#!/usr/bin/env python3
"""
Personal AI Assistant - Desktop Application
Requires Ollama to be installed and running
"""

import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import uuid
from collections import deque
import sys
import os
import subprocess
import time
import webbrowser

# Configuration
MODEL_NAME = "llama3.1:8b"  # Change this to your preferred model
APP_VERSION = "1.0.0"


def check_ollama_running():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False


def check_model_available(model_name):
    """Check if the specified model is downloaded in Ollama."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(model_name in model.get('name', '') for model in models)
        return False
    except:
        return False


def startup_checks():
    """
    Perform startup checks before launching the app.
    Returns (success, message)
    """
    print("=" * 70)
    print("PERSONAL AI ASSISTANT - STARTUP")
    print("=" * 70)
    print()
    
    # Check 1: Ollama running
    print("⏳ Checking if Ollama is running...")
    if not check_ollama_running():
        return False, """
❌ Ollama is not running!

Please start Ollama first:
1. Open Ollama application, OR
2. Run 'ollama serve' in a terminal

Then restart this application.
"""
    print("✅ Ollama is running\n")
    
    # Check 2: Model availability
    print(f"⏳ Checking if model '{MODEL_NAME}' is available...")
    if not check_model_available(MODEL_NAME):
        return False, f"""
❌ Model '{MODEL_NAME}' is not downloaded!

Please download the model first by running:
    ollama pull {MODEL_NAME}

Then restart this application.

To use a different model, edit the MODEL_NAME in the script.
"""
    print(f"✅ Model '{MODEL_NAME}' is available\n")
    
    print("✅ All checks passed! Starting application...\n")
    return True, "All systems ready!"


# Initialize AI components
print("⏳ Loading AI components...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    print("\nPlease ensure you have internet connection for first-time model download.")
    sys.exit(1)

# Get the directory where the script is located (for portable data storage)
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(APP_DIR, "assistant_data")

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="hybrid_assistant_memory",
        metadata={"description": "Hybrid memory with context window + RAG"}
    )
    print(f"✅ Database initialized at: {DB_PATH}")
    print(f"✅ Loaded {collection.count()} existing memories\n")
except Exception as e:
    print(f"❌ Error initializing database: {e}")
    sys.exit(1)

SHORT_TERM_MEMORY_SIZE = 5
short_term_memory = deque(maxlen=SHORT_TERM_MEMORY_SIZE)


def embed_text(text):
    return embedding_model.encode(text).tolist()


def store_long_term_memory(text, memory_type, metadata=None):
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
    short_term_memory.append({
        'user': user_msg,
        'assistant': assistant_msg,
        'timestamp': datetime.now().isoformat()
    })


def retrieve_long_term_memories(query, n_results=3, filter_type=None):
    if collection.count() == 0:
        return []
    
    where_filter = {"type": filter_type} if filter_type else None
    
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=min(n_results, collection.count()),
        where=where_filter
    )
    
    memories = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            memories.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
    
    return memories


def build_context_for_llm(user_query):
    context_parts = []
    
    profile_memories = collection.get(where={"type": "profile"})
    if profile_memories['documents']:
        profile_text = "USER PROFILE:\n" + "\n".join([f"- {doc}" for doc in profile_memories['documents']])
        context_parts.append(profile_text)
    
    long_term_relevant = retrieve_long_term_memories(
        user_query, 
        n_results=3,
        filter_type="conversation"
    )
    
    if long_term_relevant:
        older_memories = [m for m in long_term_relevant 
                         if len(short_term_memory) == 0 or 
                         m['metadata']['timestamp'] < short_term_memory[0]['timestamp']]
        
        if older_memories:
            lt_text = "RELEVANT PAST CONVERSATIONS:\n" + "\n".join([
                f"- {m['text']}" for m in older_memories
            ])
            context_parts.append(lt_text)
    
    if short_term_memory:
        st_text = "RECENT CONVERSATION:\n"
        for exchange in short_term_memory:
            st_text += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        context_parts.append(st_text.strip())
    
    return "\n\n".join(context_parts)


def generate_response(user_query):
    context = build_context_for_llm(user_query)
    
    prompt = f"""You are a helpful personal AI assistant with memory of your user and past conversations.

{context}

The user just said: {user_query}

Respond naturally and helpfully. Use the context to inform your response, but respond conversationally without explicitly mentioning your memory system.

Your response:"""
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': MODEL_NAME,
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"⚠️ Error: Received status code {response.status_code} from Ollama"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The model might be taking too long to respond. Try a shorter message."
    except Exception as e:
        return f"⚠️ Error connecting to Ollama: {str(e)}\n\nPlease ensure Ollama is running!"


def check_onboarding_needed():
    profile_count = len(collection.get(where={"type": "profile"})['ids'])
    return profile_count == 0


def get_onboarding_questions():
    return [
        "What are your main hobbies or interests?",
        "What do you do? (student, profession, etc.)",
        "What's your biggest strength?",
        "What are you currently working on or learning?",
        "Anything else I should know about you?"
    ]


onboarding_state = {
    'active': False,
    'current_question': 0,
    'questions': get_onboarding_questions()
}


def chat_function(message, history):
    global onboarding_state
    
    if not onboarding_state['active'] and check_onboarding_needed():
        onboarding_state['active'] = True
        onboarding_state['current_question'] = 0
        return f"Welcome! 👋 Let's get to know each other. I'll ask you a few questions.\n\nQuestion 1: {onboarding_state['questions'][0]}"
    
    if onboarding_state['active']:
        current_q = onboarding_state['questions'][onboarding_state['current_question']]
        memory_text = f"{current_q} {message}"
        store_long_term_memory(memory_text, "profile")
        
        onboarding_state['current_question'] += 1
        
        if onboarding_state['current_question'] >= len(onboarding_state['questions']):
            onboarding_state['active'] = False
            return "Thank you! ✅ I'll use this information to personalize our conversations. Feel free to start chatting!"
        else:
            next_q = onboarding_state['questions'][onboarding_state['current_question']]
            return f"Great! Next question:\n\nQuestion {onboarding_state['current_question'] + 1}: {next_q}"
    
    if not message or message.strip() == "":
        return "Please enter a message!"
    
    response = generate_response(message)
    
    add_to_short_term_memory(message, response)
    conversation_text = f"User asked: {message}\nAssistant replied: {response}"
    store_long_term_memory(conversation_text, "conversation")
    
    return response


def view_memories():
    output = "## 📚 Memory Database\n\n"
    
    profiles = collection.get(where={"type": "profile"})
    if profiles['documents']:
        output += "### 📋 Profile Information:\n"
        for doc in profiles['documents']:
            output += f"- {doc}\n"
        output += "\n"
    
    conversations = collection.get(where={"type": "conversation"})
    output += f"### 💬 Conversation History:\n"
    output += f"Total exchanges stored: {len(conversations['documents'])}\n\n"
    
    output += f"### 🧠 Short-term Buffer:\n"
    output += f"Recent exchanges in buffer: {len(short_term_memory)}/{SHORT_TERM_MEMORY_SIZE}\n"
    
    return output


def reset_memories():
    global short_term_memory, onboarding_state
    try:
        client.delete_collection("hybrid_assistant_memory")
        short_term_memory.clear()
        onboarding_state = {
            'active': False,
            'current_question': 0,
            'questions': get_onboarding_questions()
        }
        return "✅ All memories have been cleared! Please restart the application."
    except Exception as e:
        return f"❌ Error resetting memories: {str(e)}"


def export_memories():
    """Export all memories to a JSON file."""
    import json
    try:
        all_data = collection.get()
        export_data = {
            'profile': [],
            'conversations': []
        }
        
        for i, doc in enumerate(all_data['documents']):
            memory_type = all_data['metadatas'][i]['type']
            if memory_type == 'profile':
                export_data['profile'].append(doc)
            else:
                export_data['conversations'].append(doc)
        
        filename = f"memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(APP_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"✅ Memories exported to:\n{filepath}"
    except Exception as e:
        return f"❌ Error exporting memories: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="Personal AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🤖 Personal AI Assistant v{APP_VERSION}
        ### Your AI companion with memory - powered by RAG and local LLM
        
        **Model:** {MODEL_NAME} | **Status:** {"🟢 Connected" if check_ollama_running() else "🔴 Disconnected"}
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "🤖"),
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                submit = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Controls")
            
            view_memory_btn = gr.Button("📚 View Memories", size="sm")
            export_btn = gr.Button("💾 Export Memories", size="sm")
            reset_btn = gr.Button("🗑️ Reset All Memories", size="sm", variant="stop")
            
            memory_display = gr.Markdown("Click 'View Memories' to see what I know about you.")
            
            gr.Markdown(
                f"""
                ---
                ### ℹ️ System Info
                - **Data Location:** `{os.path.basename(DB_PATH)}`
                - **Model:** {MODEL_NAME}
                - **Memories:** {collection.count()}
                
                ### 💡 Tips
                - First time? I'll ask questions to learn about you
                - I remember conversations forever
                - Export memories to back them up
                
                ### ⚙️ Requirements
                - Ollama must be running
                - Model must be downloaded
                - Internet for first-time setup
                """
            )
    
    def user_submit(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot_respond(history):
        user_message = history[-1][0]
        bot_message = chat_function(user_message, history[:-1])
        history[-1][1] = bot_message
        return history
    
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    submit.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    
    view_memory_btn.click(view_memories, None, memory_display)
    export_btn.click(export_memories, None, memory_display)
    reset_btn.click(reset_memories, None, memory_display)


def main():
    """Main entry point for the application."""
    # Run startup checks
    success, message = startup_checks()
    
    if not success:
        print(message)
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
    
    # Launch the application
    print("=" * 70)
    print("🚀 LAUNCHING APPLICATION...")
    print("=" * 70)
    print()
    print("The application will open in your browser.")
    print("Keep this window open while using the app.")
    print("Press Ctrl+C to stop the application.")
    print()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        quiet=False
    )


if __name__ == "__main__":
    main()