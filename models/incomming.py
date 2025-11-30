from pydantic import BaseModel


class Incomming(BaseModel):
    message: str
    model: str



class MessageModel():
    LLMResponse: str
    UserResponse: str