from collections import deque
from datetime import datetime
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from models.incomming import MessageModel
from ragservice.IntroQuestions import IntroQuestons
import requests
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM

class LllmInterface:

    embedding_model = None
    collection = None
    PROMPTS_LOG_FILE = "./prompts_log.jsonl"
    recent_memory = None
    llm = None
    MAX_RECENT_MESSAGES = 20  # 10 Q&A pairs

    def __init__(self, sentenceTransformer = 'all-MiniLM-L6-v2', chromaDBPath = "./chroma_db_hybrid", SHORT_TERM_MEMORY_SIZE = 5):
        self.embedding_model = SentenceTransformer(sentenceTransformer)
        client = chromadb.PersistentClient(path= chromaDBPath)
        self.collection = client.get_or_create_collection(
            name="memory",
            metadata={"description": "Conversation memory with context window + RAG"}
        )

        # NEW: Add LangChain memory for onboarding
        self.recent_memory = InMemoryChatMessageHistory()

        # NEW: Initialize Ollama through LangChain with response limits
        self.llm = OllamaLLM(
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            num_predict=700,  # Limit response to ~200 tokens (~150 words)
            temperature=0.7   # Control randomness (0=deterministic, 1=creative)
        )
        

    def embed_text(self, text):
        """Converts text to embedding vector."""
        return self.embedding_model.encode(text).tolist()

    def IsOnBoardingQuetionsLeft(self):
        profile_count = len(self.collection.get(where={"type": "profile"})['ids'])
        if profile_count == len(IntroQuestons.questions) :
            return False
        else :
            return True
    
    def GetOnBoardingQuestion(self, user_response=None):
        """
        Smart onboarding flow:
        - First call (no user_response): Introduce + ask first question
        - Subsequent calls (with user_response): Acknowledge previous answer + ask next question
        """
        import json
        
        profile_count = len(self.collection.get(where={"type": "profile"})['ids'])
        
        if profile_count >= len(IntroQuestons.questions):
            return {"type": "onBoardingCompleted"}
 
        current_question = IntroQuestons.questions[profile_count][0]

        # Build prompt based on whether this is first question or follow-up
        if profile_count == 0 and user_response is None:
            # FIRST INTERACTION: Introduce yourself + ask first question
            prompt = f"""You are Julia, a helpful personal AI therapist.

Introduce yourself warmly by name, express genuine happiness to connect with this person, and then naturally ask this question: {current_question}

Be warm, conversational, and concise (3-4 sentences)."""
        
        else:
            # FOLLOW-UP: Acknowledge previous answer + ask next question
            conversation_history = self.recent_memory.messages
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history]) if conversation_history else ""
            
            # Get the last user message (their answer)
            last_answer = user_response.get("answer", "") if user_response else ""
            
            prompt = f"""You are Julia, a helpful personal AI therapist.

Previous conversation:
{history_text}

The user just answered: {last_answer}

First, briefly acknowledge and appreciate their answer (1 sentence showing you understood and care).
Then, naturally transition to ask this next question: {current_question}

Be empathetic, conversational, and concise (3-4 sentences total)."""
            
        # LangChain automatically handles context
        response = self.llm.invoke(prompt)
        
        # Store the question in memory
        self.recent_memory.add_ai_message(response)
        
        return {
            "message": response,
            "type": "onBoardingQuestions",
            "questionIndex": profile_count
        }
    def OnBoardingQuestionareComplete(self, response):
        profile_count = len(self.collection.get(where={"type": "profile"})['ids'])
        
        if profile_count == len(IntroQuestons.questions):
            # Store the last answer first
            self.recent_memory.add_user_message(response["answer"])
            
            conversation_history = self.recent_memory.messages
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history]) if conversation_history else ""
            
            # 1. Generate acknowledgment message for the user
            acknowledgment_prompt = f"""You are Julia, a helpful personal therapist.
You have previously asked different questions to know this person. Here is the previous conversation:
{history_text}

For the last question you asked, the user replied: {response["answer"]}

Now, naturally reply to this person's answer, acknowledge their sharing.
Then, let the person know that we have successfully completed the personal therapist onboarding questionnaire.
Thank them warmly for sharing this personal information as it will really help you know them better and get connected.

Be conversational, warm, and concise (4-5 sentences), and reference their previous answers if relevant."""

            acknowledgment_response = self.llm.invoke(acknowledgment_prompt)
            
            # 2. Generate internal summary for RAG (this won't be shown to user)
            summary_prompt = f"""You are a clinical psychologist analyzing a therapy intake session. Based on the following Q&A conversation, create a comprehensive psychological profile summary.

Complete conversation:
{history_text}

Create a structured clinical summary covering:

1. **Presenting Issues**: What brought them here? What are their primary concerns and struggles?
2. **Emotional State**: Current mental/emotional wellbeing, stress levels, mood patterns
3. **Triggers & Patterns**: Recurring situations, people, or thoughts that cause distress
4. **Coping Mechanisms**: How they currently handle stress (healthy vs. unhealthy patterns)
5. **Strengths & Resources**: Personal strengths, positive moments, support system, resilience factors
6. **Goals & Motivation**: What they want to change, their hopes for therapy, desired outcomes
7. **Risk Factors**: Any concerning patterns, isolation, harmful coping strategies (mention if none detected)
8. **Therapeutic Approach**: Recommended focus areas and intervention strategies for future sessions

Be analytical, clinically informed, and comprehensive (10-12 sentences). Use empathetic but professional language. This profile will guide all future therapeutic interactions."""

            summary = self.llm.invoke(summary_prompt)
            
            # 3. Store the summary in ChromaDB for future RAG retrieval
            self.store_long_term_memory(
                text=summary,
                memory_type="summary",
                metadata={
                    "summary_type": "onboarding_profile",
                    "question_count": len(IntroQuestons.questions)
                }
            )
            
            # 4. Store the acknowledgment in memory
            self.recent_memory.add_ai_message(acknowledgment_response)

            return {
                "message": acknowledgment_response,
                "type": "onBoardingCompleted",
            }


    def StoreOnboardingQuestionare(self, response):
        question = response["question"]
        answer = response["answer"]
        
        # AI question already stored in GetOnBoardingQuestion()
        # Only store user's answer
        self.recent_memory.add_user_message(answer)
        
        # Store in ChromaDB (permanent storage)
        conversation = f"Question: {question}\nAnswer: {answer}"
        return self.store_long_term_memory(conversation, "profile")

    def store_conversation(self, user_message, ai_response):
        # Store in recent memory
        self.recent_memory.add_user_message(user_message)
        self.recent_memory.add_ai_message(ai_response)
        
        # Trim to last 10 conversations
        if len(self.recent_memory.messages) > self.MAX_RECENT_MESSAGES:
            self.recent_memory.messages = self.recent_memory.messages[-self.MAX_RECENT_MESSAGES:]
        
        # Store in ChromaDB (permanent)
        conversation = f"User: {user_message}\nAI: {ai_response}"
        self.store_long_term_memory(conversation, "chat")
    
    def get_ai_response(self, user_message):
        # Get recent context (last 10)
        recent_context = self.recent_memory.messages
        
        # Format recent messages into readable text
        recent_text = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_context]) if recent_context else "No recent conversation"
        
        # RAG: Find relevant old conversations
        relevant_memories = self.collection.query(
            query_embeddings=[self.embed_text(user_message)],
            n_results=3,  # Top 3 relevant memories
            where={"type": {"$in": ["chat", "profile"]}}
        )
        
        # Format relevant memories
        if relevant_memories['documents'] and relevant_memories['documents'][0]:
            memories_text = "\n\n".join(relevant_memories['documents'][0])
        else:
            memories_text = "No relevant past memories"
        
        # Build prompt
        prompt = f"""You are Julia, the user's personal AI assistant.
        
        Recent conversation:
        {recent_text}
        
        Relevant memories from past:
        {memories_text}
        
        User: {user_message}
        
        Respond naturally, concisely (2-3 sentences), and reference past conversations when relevant."""
        
        response = self.llm.invoke(prompt)
        
        # Store this exchange
        self.store_conversation(user_message, response)
        
        return response
    
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