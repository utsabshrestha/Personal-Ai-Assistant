import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
import uuid

# Initialize the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!\n")

# Initialize ChromaDB with persistent storage
# This creates a folder called 'chroma_db' where all your data will be saved
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection (like a table in a database)
# A collection holds all your embeddings and their associated text
collection = client.get_or_create_collection(
    name="personal_assistant_memory",
    metadata={"description": "Stores user profile and conversation history"}
)

print(f"Memory database loaded. Currently storing {collection.count()} memories.\n")


def embed_text(text):
    """
    Converts text into an embedding vector.
    We use this same function for both storing and searching.
    """
    return embedding_model.encode(text).tolist()


def store_memory(text, memory_type, metadata=None):
    """
    Stores a piece of information in the memory database.
    
    text: The actual text content to store
    memory_type: Category like 'profile', 'conversation', 'user_message', 'assistant_message'
    metadata: Any additional information you want to store with this memory
    """
    # Create a unique ID for this memory
    memory_id = str(uuid.uuid4())
    
    # Prepare metadata
    meta = {
        "type": memory_type,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)
    
    # Store in ChromaDB
    # ChromaDB automatically handles the embedding if we provide the embedding function,
    # but we'll do it manually to have more control
    collection.add(
        ids=[memory_id],
        embeddings=[embed_text(text)],
        documents=[text],
        metadatas=[meta]
    )
    
    return memory_id


def retrieve_relevant_memories(query, n_results=5, filter_type=None):
    """
    Searches through all stored memories to find the most relevant ones.
    
    query: What to search for
    n_results: How many relevant memories to return
    filter_type: Optional filter to only search specific types of memories
    """
    # Prepare filter if specified
    where_filter = {"type": filter_type} if filter_type else None
    
    # Query ChromaDB - it automatically calculates similarity and returns top results
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=min(n_results, collection.count()),  # Don't request more than we have
        where=where_filter
    )
    
    # ChromaDB returns results in a specific format, let's make it easier to work with
    memories = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            memories.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]  # Lower distance = more similar
            })
    
    return memories


def generate_response(query, relevant_memories):
    """
    Sends the query and retrieved memories to Ollama for response generation.
    """
    # Separate profile info from conversational context for better prompt structure
    profile_memories = [m for m in relevant_memories if m['metadata']['type'] == 'profile']
    conversation_memories = [m for m in relevant_memories if m['metadata']['type'] == 'conversation']
    
    # Build context sections
    profile_context = ""
    if profile_memories:
        profile_context = "What you know about the user:\n" + "\n".join([m['text'] for m in profile_memories])
    
    conversation_context = ""
    if conversation_memories:
        conversation_context = "Relevant past conversations:\n" + "\n".join([m['text'] for m in conversation_memories])
    
    # Construct prompt with structured context
    prompt = f"""You are a helpful personal AI assistant. You have access to information about the user and your past conversations.

{profile_context}

{conversation_context}

Based on this context, please respond naturally to the user's message. Use the context to personalize your response, but don't explicitly mention that you're "looking at your notes" or "checking my memory" - just respond as if you naturally remember these things.

User's message: {query}

Your response:"""
    
    # Call Ollama
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2:3b',
            'prompt': prompt,
            'stream': False,
            "options": {
                "num_predict": 8192 
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['response']
    else:
        return f"Error: Could not generate response. Status code: {response.status_code}"


def onboarding_phase():
    """
    Initial onboarding to learn about the user.
    Only runs if there are no profile memories stored yet.
    """
    # Check if we already have profile information
    profile_count = len(collection.get(where={"type": "profile"})['ids'])
    
    if profile_count > 0:
        print(f"Welcome back! I already know {profile_count} things about you.\n")
        return False
    
    print("=" * 60)
    print("WELCOME! Let's get to know each other.")
    print("=" * 60)
    print("I'd like to learn about you so I can be a better assistant.")
    print("Please answer the following questions:\n")
    
    questions = [
        "What are your main hobbies or interests?",
        "Can you describe your lifestyle? (e.g., student, working professional, etc.)",
        "What would you say is your biggest strength?",
        "What's something you're currently working on or interested in learning?",
        "Is there anything else important I should know about you?"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        answer = input("Your answer: ").strip()
        
        if answer:
            # Store both the question and answer together as context
            memory_text = f"User's {question.lower()}: {answer}"
            store_memory(memory_text, "profile")
            print("  ✓ Remembered!\n")
        else:
            print("  (Skipped)\n")
    
    print("=" * 60)
    print("Thank you! I'll use this information to personalize our conversations.")
    print("=" * 60)
    print()
    
    return True


def chat_loop():
    """
    Main conversational loop with memory integration.
    """
    print("You can start chatting now! (Type 'quit' to exit, 'memories' to see what I know)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! I'll remember our conversation for next time.")
            break
        
        # Special command to view stored memories
        if user_input.lower() == 'memories':
            show_memories()
            continue
        
        # Special command to reset everything
        if user_input.lower() == 'reset':
            confirm = input("Are you sure you want to delete all memories? (yes/no): ")
            if confirm.lower() == 'yes':
                client.delete_collection("personal_assistant_memory")
                print("All memories deleted. Please restart the program.\n")
                break
            continue
        
        print()  # Blank line for readability
        
        # Retrieve relevant memories for this query
        # We retrieve both profile info and past conversations
        relevant_memories = retrieve_relevant_memories(user_input, n_results=5)
        
        # Show what context we're using (helpful for understanding the system)
        if relevant_memories:
            print("(Thinking about relevant context...)")
        
        # Generate response
        response = generate_response(user_input, relevant_memories)
        
        print(f"\nAssistant: {response}\n")
        
        # Store this exchange in memory for future reference
        # We store it as a single conversational unit
        conversation_text = f"User said: {user_input}\nAssistant responded: {response}"
        store_memory(conversation_text, "conversation")
        
        print("-" * 60)


def show_memories():
    """
    Displays all stored memories organized by type.
    """
    print("\n" + "=" * 60)
    print("STORED MEMORIES")
    print("=" * 60)
    
    # Get all profile memories
    profiles = collection.get(where={"type": "profile"})
    if profiles['documents']:
        print("\n📋 PROFILE INFORMATION:")
        for doc in profiles['documents']:
            print(f"  • {doc}")
    
    # Get conversation count
    conversations = collection.get(where={"type": "conversation"})
    if conversations['documents']:
        print(f"\n💬 CONVERSATIONS: {len(conversations['documents'])} exchanges stored")
        print("\nMost recent conversations:")
        # Show last 3 conversations
        for doc in conversations['documents'][-3:]:
            print(f"\n  {doc}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    print("=" * 60)
    print("PERSONAL AI ASSISTANT WITH CONVERSATIONAL MEMORY")
    print("=" * 60)
    print()
    
    # Run onboarding if this is the first time
    is_first_time = onboarding_phase()
    
    if not is_first_time:
        print("Type 'memories' at any time to see what I know about you.")
        print()
    
    # Start the chat loop
    chat_loop()


if __name__ == "__main__":
    main()