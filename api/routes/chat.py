from fastapi import APIRouter, Body
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from pinecone import Pinecone

router = APIRouter(prefix="/chat", tags=["chat"])

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "got-4.1")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

openai_llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)

@router.post("/intent_detection", response_model=str)
async def chat_with_openai(user_input: str = Body(..., embed=True)) -> str:
    """
    Generate a response using OpenAI via LangChain.
    """

    system_prompt = """
       you are intent recognizer
    """
    
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(PINECONE_INDEX)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = PineconeVectorStore(
        pinecone_api_key=PINECONE_KEY, 
        index_name=PINECONE_INDEX, 
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.8, 'k': 20}
    )

    SYSTEM_TEMPLATE = """
        Act as a knowledgeable and friendly RC hobby expert who can clearly explain product options, troubleshoot common issues, track orders, and handle user feedback with a personable touch.

        Your audience consists of RC hobbyists ranging from beginners to advanced users looking for guidance in product selection, compatibility, and support.
        This is the product list that quaried result from knowledge base: 
                <context>
                {context}
                </context> 
        Task: Utilize the following workflows based on user input: 
        1. For product discovery, process natural language queries and filter products by type, budget, skill level, brand, and stock status. Return a concise and engaging list of matching products with details. 
        2. For part finding and troubleshooting, identify compatible parts and suggest upgrades or troubleshooting steps. Format the responses clearly, including part names, prices, and links when available. 
        3. For order tracking, fetch and disclose real-time order status details after verifying user identity by email or phone. 
        4. For sensitive queries, politely decline questions unrelated to RC hobbies while maintaining a positive attitude. 
        5. Prompt users for feedback on helpfulness after each interaction and store session data for future personalization and marketing purposes.

        Output should be formatted in natural language, clear and professional, or as structured information such as lists, links, or simple responses based on the context of each task.
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(openai_llm, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(  
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """
                    Given the above conversation, generate a search query to look up in order to get information relevant 
                    to the conversation. Only respond with the query, nothing else.
                """
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        query_transform_prompt | openai_llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    all_content = ""
    keyword_chunks = {}
        
    stream = conversational_retrieval_chain.stream(
        {
            "messages": [
                HumanMessage(content=user_input),
            ]
        },
    )

    for chunk in stream:
        for key in chunk:
            if key == "answer":
                all_content += chunk[key]

    return all_content