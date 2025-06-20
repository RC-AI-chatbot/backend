from fastapi import APIRouter, Body
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

router = APIRouter(prefix="/chat", tags=["chat"])

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "got-4.1")

openai_llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

@router.post("/intent_detection", response_model=str)
async def chat_with_openai(user_input: str = Body(..., embed=True)) -> str:
    """
    Generate a response using OpenAI via LangChain.
    """

    system_prompt = """
       you are intent recognizer
    """

    messages = [
        ("system", system_prompt),
        ("user", user_input),
    ]
    ai_response = openai_llm.invoke(messages).content

    return ai_response