import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# STEP 1: Load the embedding model
# This model converts text into numerical vectors that capture meaning
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!\n")

# STEP 2: Load and chunk your documents
def load_and_chunk_documents(file_path, chunk_size=500):
    """
    Reads a text file and splits it into manageable chunks.
    
    chunk_size: approximate number of characters per chunk
    We overlap chunks slightly to avoid cutting sentences awkwardly
    """
    print(f"Loading documents from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple chunking: split by paragraphs first, then combine into chunks
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk_size, save current chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Created {len(chunks)} chunks from your documents.\n")
    return chunks

# STEP 3: Create embeddings for all chunks
def create_embeddings(chunks):
    """
    Converts each text chunk into a numerical vector (embedding).
    These embeddings capture the semantic meaning of the text.
    """
    print("Creating embeddings for all chunks...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    print(f"Created {len(embeddings)} embeddings.\n")
    return embeddings

# STEP 4: Retrieve relevant chunks based on a query
def retrieve_relevant_chunks(query, chunks, embeddings, top_k=3):
    """
    Finds the most relevant chunks for a given query.
    
    1. Convert query to embedding
    2. Calculate similarity between query and all chunk embeddings
    3. Return the top_k most similar chunks
    """
    print(f"Searching for relevant information about: '{query}'")
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    
    # Calculate cosine similarity between query and all chunks
    # Cosine similarity measures how "aligned" two vectors are (ranges from -1 to 1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Retrieve the actual chunks and their similarity scores
    relevant_chunks = []
    for idx in top_indices:
        relevant_chunks.append({
            'text': chunks[idx],
            'similarity': similarities[idx]
        })
        print(f"  - Found relevant chunk (similarity: {similarities[idx]:.3f})")
    
    print()
    return relevant_chunks

# STEP 5: Generate response using Ollama
def generate_response(query, relevant_chunks):
    """
    Sends the query and retrieved context to Ollama for response generation.
    """
    # Construct the context from retrieved chunks
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Create a prompt that includes both context and query
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    print("Prompt Passed : " + prompt)
    # print("Generating response from Ollama...")
    
    # Call Ollama API (running locally on your machine)
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2:3b',
            'prompt': prompt,
            'stream': False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['response']
    else:
        return f"Error: Could not generate response. Status code: {response.status_code}"

# MAIN EXECUTION
def main():
    print("=" * 60)
    print("SIMPLE RAG SYSTEM - FROM SCRATCH")
    print("=" * 60)
    print()
    
    # Load and prepare your documents
    chunks = load_and_chunk_documents('documents.txt')
    
    # Create embeddings for all chunks (this is your "vector database")
    embeddings = create_embeddings(chunks)
    
    print("=" * 60)
    print("RAG system is ready! You can now ask questions.")
    print("=" * 60)
    print()
    
    # Interactive query loop
    while True:
        query = input("Enter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n" + "-" * 60)
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, top_k=3)
        
        # Generate response
        response = generate_response(query, relevant_chunks)
        
        print("\nRESPONSE:")
        print(response)
        print("-" * 60)
        print()

if __name__ == "__main__":
    main()