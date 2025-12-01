from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from models.incomming import Incomming
from ragservice.LllmInterface import LllmInterface

app  = FastAPI()

# Create an instance of LllmInterface
llm_interface = LllmInterface()


@app.post("/send")
async def message(incomming: Incomming):
    return {"message" : incomming.message, "model": incomming.model}


@app.websocket("/messaging")
async def MessageSocketEndpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive as text first to debug
            text = await websocket.receive_text()
            print(f"Received raw text: {text}")
            
            # Try to parse JSON
            try:
                import json
                response = json.loads(text)
            except json.JSONDecodeError as e:
                await websocket.send_json({"error": f"Invalid JSON: {str(e)}", "received": text})
                continue
            
            if response['type'] == 'newConnection':
                # First question - introduce + ask (no user response yet)
                question = llm_interface.GetOnBoardingQuestion(user_response=None)
                await websocket.send_json(question)
                
            if response['type'] == 'onBoardingResponse':
                # Store answer (adds to memory)
                llm_interface.StoreOnboardingQuestionare(response)
                
                # Get next question WITH acknowledgment of previous answer
                next_question = llm_interface.GetOnBoardingQuestion(user_response=response)
                if next_question['type'] == 'onBoardingCompleted':
                    llm_response = llm_interface.OnBoardingQuestionareComplete(response)
                    await websocket.send_json(llm_response)
                else:
                    await websocket.send_json(next_question)

            elif response['type'] == 'ConversationOn':
                llm_response = llm_interface.get_ai_response(response['question'])
                await websocket.send_json(response)
                


    except WebSocketDisconnect:
        print("client disconnected")


## question -> question + answer, LLM(question + anser + new question) -> question + answer, LLM(question + answer + new question)