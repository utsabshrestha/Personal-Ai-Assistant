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
            response = await websocket.receive_json()
            if response['type'] == 'newConnection':
                # First question - no context yet
                question = llm_interface.GetOnBoardingQuestion()
                await websocket.send_json(question)
                
            elif response['type'] == 'onBoardingResponse':
                # Store answer (adds to memory)
                llm_interface.StoreOnboardingQuestionare(response)
                
                # Get next question WITH CONTEXT of previous answers
                next_question = llm_interface.GetOnBoardingQuestion()
                await websocket.send_json(next_question)

    except WebSocketDisconnect:
        print("client disconnected")


## question -> question + answer, LLM(question + anser + new question) -> question + answer, LLM(question + answer + new question)