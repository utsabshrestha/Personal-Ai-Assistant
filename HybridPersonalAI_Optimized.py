import requests
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import uuid
from collections import deque
import os
import json

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db_hybrid")
collection = client.get_or_create_collection(
    name="hybrid_assistant_memory",
    metadata={"description": "Hybrid memory with context window + RAG"}
)

print(f"Memory database loaded. {collection.count()} long-term memories stored.\n")

# This is our short-term memory buffer
# OPTIMIZED: Reduced from 7 to 5 for faster context processing
SHORT_TERM_MEMORY_SIZE = 5  # Keep last 5 exchanges in full context
short_term_memory = deque(maxlen=SHORT_TERM_MEMORY_SIZE)

# Prompt logging configuration
PROMPTS_LOG_FILE = "./prompts_log.jsonl"  # JSONL format (one JSON object per line)


def log_prompt_to_file(user_query, prompt, response):
    """
    Logs the prompt sent to LLM with timestamp to a file.
    Format: JSONL (JSON Lines) for easy parsing.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "prompt_sent_to_llm": prompt,
        "llm_response": response
    }
    
    try:
        with open(PROMPTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"⚠️  Error logging prompt: {e}")


def embed_text(text):
    """Converts text to embedding vector."""
    return embedding_model.encode(text).tolist()


def store_long_term_memory(text, memory_type, metadata=None):
    """
    Stores information in long-term memory (ChromaDB).
    This is for memories that should be searchable across sessions.
    """
    memory_id = str(uuid.uuid4())
    meta = {
        "type": memory_type,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)
    
    collection.add(
        ids=[memory_id],
        embeddings=[embed_text(text)],
        documents=[text],
        metadatas=[meta]
    )
    
    return memory_id


def add_to_short_term_memory(user_msg, assistant_msg):
    """
    Adds an exchange to short-term memory buffer.
    This keeps recent conversation in full context for narrative continuity.
    """
    short_term_memory.append({
        'user': user_msg,
        'assistant': assistant_msg,
        'timestamp': datetime.now().isoformat()
    })


def build_context_for_llm(user_query):
    """
    OPTIMIZED: Builds context with performance improvements:
    1. Cache query embedding (avoid duplicate encoding)
    2. Reduce ChromaDB queries and result counts
    3. Stricter distance threshold (1.0 instead of 1.5)
    4. Avoid expensive collection.get() for all conversations
    """
    context_parts = []
    
    # OPTIMIZATION: Embed query once and reuse for all searches
    query_embedding = embed_text(user_query)
    
    # PART 1: Summary Profile (always include, no retrieval needed)
    profile_summary = collection.get(where={"type": "summary"})
    if profile_summary['documents']:
        # Only use first summary to keep context concise
        profile_text = "USER SUMMARY:\n" + profile_summary['documents'][0]
        context_parts.append(profile_text)
    
    # PART 2: Relevant profile details (OPTIMIZED: top 2 only, using cached embedding)
    profile_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(2, collection.count()),
        where={"type": "profile"}
    )
    
    if profile_results['documents'] and profile_results['documents'][0]:
        # Filter by stricter distance threshold (1.0 instead of 1.5)
        relevant_profiles = [
            profile_results['documents'][0][i] 
            for i in range(len(profile_results['documents'][0])) 
            if profile_results['distances'][0][i] <= 1.0
        ]
        if relevant_profiles:
            profile_text = "RELEVANT PROFILE:\n" + "\n".join([f"- {doc}" for doc in relevant_profiles])
            context_parts.append(profile_text)
    
    # PART 3: Relevant past conversations (OPTIMIZED: top 2 only, using cached embedding)
    convo_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(2, collection.count()),
        where={"type": "conversation"}
    )
    
    if convo_results['documents'] and convo_results['documents'][0]:
        # Filter by stricter distance threshold
        relevant_convos = []
        for i in range(len(convo_results['documents'][0])):
            if convo_results['distances'][0][i] <= 1.0:
                # Filter out very recent conversations (already in short-term memory)
                if len(short_term_memory) == 0 or convo_results['metadatas'][0][i]['timestamp'] < short_term_memory[0]['timestamp']:
                    relevant_convos.append(convo_results['documents'][0][i])
        
        if relevant_convos:
            lt_text = "PAST CONVERSATIONS:\n" + "\n".join([f"- {doc}" for doc in relevant_convos])
            context_parts.append(lt_text)
    
    # PART 4: Short-term memory (recent conversation in full)
    if short_term_memory:
        st_text = "RECENT CONVERSATION:\n"
        for exchange in short_term_memory:
            st_text += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        context_parts.append(st_text.strip())
    else:
        # OPTIMIZED: Use semantic query instead of fetching all conversations
        # This scales much better as database grows
        recent_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, collection.count()),
            where={"type": "conversation"}
        )
        
        if recent_results['documents'] and recent_results['documents'][0]:
            prev_text = "RECENT PAST CONVERSATIONS:\n"
            for doc in recent_results['documents'][0][:3]:
                prev_text += f"- {doc}\n"
            context_parts.append(prev_text.strip())
    
    return "\n\n".join(context_parts)


def get_response(prompt):
    """Helper function to call Ollama LLM."""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.1:8b',
            'prompt': prompt,
            'stream': False
        }
    )
    
    if response.status_code == 200:
        llm_response = response.json()['response']
        return llm_response
    else:
        error_msg = f"Error: Status {response.status_code}"
        return error_msg

    
def generate_response(user_query):
    """
    Generates a response using hybrid memory approach.
    """
    # Build comprehensive context
    context = build_context_for_llm(user_query)
    
    # Construct the prompt
    prompt = f"""You are a helpful personal AI assistant with memory of your user and past conversations.

{context}

The user just said: {user_query}

Respond naturally and helpfully. Use the context to inform your response, but respond conversationally without explicitly mentioning your memory system.

Your response:"""
    
    # Call Ollama
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.1:8b',
            'prompt': prompt,
            'stream': False
        }
    )
    
    if response.status_code == 200:
        llm_response = response.json()['response']
        # Log the prompt and response to file with timestamp
        log_prompt_to_file(user_query, prompt, llm_response)
        return llm_response
    else:
        error_msg = f"Error: Status {response.status_code}"
        log_prompt_to_file(user_query, prompt, error_msg)
        return error_msg


def onboarding_phase():
    """Initial onboarding to collect user profile."""
    profile_count = len(collection.get(where={"type": "profile"})['ids'])
    
    if profile_count > 0:
        print(f"Welcome back! I remember you. 😊\n")
        return False
    
    print("=" * 60)
    print("WELCOME! Let's get to know each other.")
    print("=" * 60)
    print()
    
    print(f"Q: Before starting the conversation, what should i call you my friend ? 😊 ")
    name = input("A: ").strip()
    store_long_term_memory(f"You can call this person by a name : '{name}'.", "profile")

    questions = [
        ["So...  what are you currently working on or learning?","What does this person is currently working or learning currently?"],
        ["And what do you do, and what do you love about it?", "what does this person do, and what does this person loves about it?"],
        ["I'm curious... what are your main hobbies or interests outside of work?","what are the main hobbies or interests outside of the work of this person ?"],
        ["Hmm... what's a challenge you're facing right now?", "What challenges does this person is facing right now ?"],
        ["Out of curiosity... how do you like to spend your ideal day?","How does this person like to spend their ideal day?"]
    ]
    
    AllQuestionsAndAnser = f"""This is a question and answer asked to a person to know them personally. 
    You have to summerize the nature of the person by analyzing these questions and answer. 
    Your response should be in a concise way detailly analyzing the real nature, strengths, weakness, behavious, life goals of this person. 
    Provide response in a paragraph only.\n
    """

    for question in questions:
        print(f"Q: {question[0]}")
        answer = input("A: ").strip()
        
        if answer:
            memory_text = f"{question[1]} Answer : {answer}"
            store_long_term_memory(memory_text, "profile")
            AllQuestionsAndAnser += f"Question : {question[1]} \n Answer : {answer}"
            print("  ✓ Remembered!\n")
        else:
            print("  (Skipped)\n")
    
    if AllQuestionsAndAnser:
        response = get_response(AllQuestionsAndAnser)
        if response:
            store_long_term_memory(response, "summary")
    
    print("=" * 60)
    print("Thanks! Let's chat.")
    print("=" * 60)
    print()
    return True


def show_memory_status():
    """Shows current memory state."""
    print("\n" + "=" * 60)
    print("MEMORY STATUS")
    print("=" * 60)
    
    # Profile memories
    profiles = collection.get(where={"type": "profile"})
    print(f"\n📋 Profile Facts: {len(profiles['documents'])}")
    for doc in profiles['documents']:
        print(f"  • {doc}")
    
    # Long-term conversation count
    conversations = collection.get(where={"type": "conversation"})
    print(f"\n💾 Long-term Memories: {len(conversations['documents'])} conversations")
    
    # Short-term memory
    print(f"\n🧠 Short-term Buffer: {len(short_term_memory)}/{SHORT_TERM_MEMORY_SIZE} recent exchanges")
    if short_term_memory:
        print("\nRecent conversation in buffer:")
        for i, exchange in enumerate(short_term_memory, 1):
            print(f"\n  Exchange {i}:")
            print(f"    You: {exchange['user'][:60]}...")
            print(f"    AI: {exchange['assistant'][:60]}...")
    
    print("\n" + "=" * 60 + "\n")


def chat_loop():
    """Main chat loop with hybrid memory."""
    print("Chat commands: 'quit' to exit, 'memory' to view status, 'reset' to clear all\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! I'll remember our conversation.")
            break
        
        if user_input.lower() == 'memory':
            show_memory_status()
            continue
        
        if user_input.lower() == 'reset':
            confirm = input("Delete ALL memories? (yes/no): ")
            if confirm.lower() == 'yes':
                client.delete_collection("hybrid_assistant_memory")
                short_term_memory.clear()
                print("All memories cleared. Please restart.\n")
                break
            continue
        
        print()
        
        # Generate response using hybrid memory
        response = generate_response(user_input)
        
        print(f"Assistant: {response}\n")
        print("-" * 60)
        
        # Store in BOTH memory systems:
        
        # 1. Add to short-term memory buffer (always kept in full)
        add_to_short_term_memory(user_input, response)
        
        # 2. Store in long-term memory (for retrieval in future sessions)
        conversation_text = f"User asked: {user_input}\nAssistant replied: {response}"
        store_long_term_memory(conversation_text, "conversation")


def main():
    print("=" * 60)
    print("HYBRID MEMORY AI ASSISTANT (OPTIMIZED)")
    print("Context Window + RAG Combined")
    print("=" * 60)
    print()
    
    onboarding_phase()
    chat_loop()


if __name__ == "__main__":
    main()
