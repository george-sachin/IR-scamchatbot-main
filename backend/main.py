from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Chatbot import Chatbot
from pydantic import BaseModel
from typing import Union

class Item(BaseModel):
    text: str
    topic: Union[str, None] = None 

cb = Chatbot()
print("Chatbot Loaded")
app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/next_response')
def get_response(item: Item):
    response = cb.generate_response(item.text, item.topic)
    return {
        "response": response,
    }

@app.put('/reset')
def reset_chatbot():
    cb.reset_conv()
    return {
        "reset": True
    }