from collections import deque
from datetime import datetime
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from models.incomming import MessageModel
from ragservice.IntroQuestions import IntroQuestons
import requests
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

class LllmInterface:

    embedding_model = None
    collection = None
    short_term_memory = None
    PROMPTS_LOG_FILE = "./prompts_log.jsonl"

    def __init__(self, sentenceTransformer = 'all-MiniLM-L6-v2', chromaDBPath = "../chroma_db_hybrid", SHORT_TERM_MEMORY_SIZE = 5):
        self.embedding_model = SentenceTransformer(sentenceTransformer)
        client = chromadb.PersistentClient(path= chromaDBPath)
        self.collection = client.get_or_create_collection(
            name="hybrid_assistant_memory",
            metadata={"description": "Hybrid memory with context window + RAG"}
        )
        self.short_term_memory = deque(maxlen=SHORT_TERM_MEMORY_SIZE)

        # NEW: Add LangChain memory for onboarding
        self.onboarding_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # NEW: Initialize Ollama through LangChain
        self.llm = Ollama(
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        )
        

    def embed_text(self, text):
        """Converts text to embedding vector."""
        return self.embedding_model.encode(text).tolist()

    def isNewOnBoarding(self):
        profile_count = len(self.collection.get(where={"type": "profile"})['ids'])
        if profile_count > 0 :
            return True
        else :
            return False
    
    def GetOnBoardingQuestion(self):
        import json
        
        profile_count = len(self.collection.get(where={"type": "profile"})['ids'])
        
        if profile_count >= len(IntroQuestons.questions):
            return {"type": "onBoardingCompleted"}
 
        current_question = IntroQuestons.questions[profile_count][0]

        # Build prompt WITH previous conversation context
        conversation_history = self.onboarding_memory.load_memory_variables({})

        prompt = f"""You are Julia, a helpful personal AI assistant.

        Previous conversation:
        {conversation_history.get('chat_history', 'This is the first question')}

        Now, naturally ask this question: {current_question}

        Be conversational and reference their previous answers if relevant."""
            
        # LangChain automatically handles context
        response = self.llm.invoke(prompt)
        
        # Store the question in memory
        self.onboarding_memory.chat_memory.add_ai_message(response)
        
        return {
            "message": response,
            "type": "onBoardingQuestions",
            "questionIndex": profile_count
        }
        
    def StoreOnboardingQuestionare(self, response):
        question = response["question"]
        answer = response["answer"]
        
        # Store AI question first (if not already stored)
        self.onboarding_memory.chat_memory.add_ai_message(question)
        
        # Then store user's answer
        self.onboarding_memory.chat_memory.add_user_message(answer)
        
        # Store in ChromaDB
        conversation = f"Question: {question}\nAnswer: {answer}"
        return self.store_long_term_memory(conversation, "profile")       
        
    
    def isOnBoardingQuestionsLeft(self):
        questionsAsked = len(self.collection.get(where={"type":"profile"})['ids'])
        if questionsAsked < len(IntroQuestons.questions):
            return True
        else:
            return False

            
    def store_long_term_memory(self, text, memory_type, metadata=None):
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
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[self.embed_text(text)],
            documents=[text],
            metadatas=[meta]
        )
        
        return memory_id