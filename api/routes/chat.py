from fastapi import APIRouter, Body
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import typesense

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

from core.prompt import INTENT_DETECTION_SYSTEM_PROMPT

router = APIRouter(prefix="/chat", tags=["chat"])

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "got-4.1")
PINECONE_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "")
TYPESENSE_PORT = os.getenv("TYPESENSE_PORT", "")

openai_llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

client = typesense.Client({
        "nodes": [{
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": "https"
    }],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 1000
})

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)

class ChatRequest(BaseModel):
    user_input: str
    history: str

@router.post("/chat_with_openai", response_model=str)
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

@router.post("/chat_with_typesense", response_model=str)
async def chat_with_typesense(user_input: str = Body(..., embed=True)) -> str:
    """
    AI chat using Typesense as retrieval backend (RAG style).
    """
    # 1. Search Typesense for relevant products
    search_parameters = {
        "q": user_input,
        "query_by": "title,description,brand,sku",
        "per_page": 5
    }
    try:
        search_results = client.collections['slot_cars'].documents.search(search_parameters)
        hits = search_results.get("hits", [])
        if not hits:
            context = "No relevant products found."
        else:
            # Build a context string from the top results
            context = "\n\n".join(
                f"Title: {hit['document'].get('title', '')}\n"
                f"Description: {hit['document'].get('description', '')}\n"
                f"Brand: {hit['document'].get('brand', '')}\n"
                f"Price: {hit['document'].get('price', '')}\n"
                f"Link: {hit['document'].get('link', '')}"
                for hit in hits
            )
    except Exception as e:
        context = f"Error searching Typesense: {e}"

    # 2. Compose a simple prompt for the LLM
    prompt = f"""
        You are an RC product assistant. Use the following product information to help the user:

        {context}

        User question: {user_input}

        Answer:
    """

    # 3. Get LLM response
    response = openai_llm.invoke(prompt)

    return response.content

@router.post("/chat_with_ai", response_model=str)
async def chat_with_ai(request: ChatRequest):
    """
    AI chat using Typesense as retrieval backend (RAG style).
    """
    user_input = request.user_input
    history = request.history
    # Detect user's input as keyword
    detect_system_prompt = INTENT_DETECTION_SYSTEM_PROMPT

    # Compose the prompt for the LLM
    detect_prompt = f"{detect_system_prompt}\n\nUser input: {user_input}\nIntent:"

    # Get LLM response as keyword
    keyword_response = openai_llm.invoke(detect_system_prompt)
    keyword = keyword_response.content.strip() if hasattr(keyword_response, "content") else str(keyword_response)
    
    print("Keyword: ", keyword)

    if keyword == "product" or keyword == "response":
        # 2. Search Typesense for relevant products
        search_parameters = {
            "q": user_input,
            "query_by": "title,description,brand,sku",
            "per_page": 5
        }
        try:
            search_results = client.collections['slot_cars'].documents.search(search_parameters)
            hits = search_results.get("hits", [])
            if not hits:
                context = "No relevant products found."
            else:
                # Build a context string from the top results
                context = "\n\n".join(
                    f"Title: {hit['document'].get('title', '')}\n"
                    f"Description: {hit['document'].get('description', '')}\n"
                    f"Brand: {hit['document'].get('brand', '')}\n"
                    f"Price: {hit['document'].get('price', '')}\n"
                    f"Link: {hit['document'].get('link', '')}\n"
                    f"Link: {hit['document'].get('image_link', '')}"
                    for hit in hits
                )
        except Exception as e:
            context = f"Error searching Typesense: {e}"

        # 2. Compose a simple prompt for the LLM
        prompt_response_product = f"""
            You are a highly skilled RC product assistant with expertise in remote control vehicles and parts. Analyze the user’s question and generate a clear, concise, and helpful response. Your answer should directly address the user’s intent, using only the provided product data.

            Instructions:

            If the question requests product recommendations or lists, select the most relevant products using the following criteria (unless otherwise specified by the user):
                Default sort: total_sales (highest first), and In Stock = Yes.
                If price constraints are given: Only include products where (price or sale_price) is within the specified range.
                If specific types, categories, scales, or brands are mentioned: Only show products matching those requirements (e.g., “1/10 scale”, “Traxxas”, “RPM”, “MIP”, “off-road”, “drift”, etc.).
                If compatibility or upgrade requests: Only show parts or upgrades specifically compatible with the mentioned model.
                If user requests beginner/intermediate/advanced: Select products with a corresponding skill level tag, or use product descriptions to infer suitability.
            Product listing rules:
                Show only 3 products (unique parent_sku) unless the user asks to see more. Never show more than 5 products.
                Always format product results in a user-friendly list, including: Product Name, Short Description (if available), Price, and a short note on why it’s recommended for the query.
                If relevant, group products by category (e.g., “Wheels”, “Motors”, “Body Shells”) before listing.
            If you cannot find relevant products, explain why and provide helpful suggestions or alternatives.
            For technical, compatibility, or installation questions, provide a concise, factual answer based only on the data. If you can list compatible products or parts, do so.
            Do not list products outside of the given context or product categories.
            Never recommend slot cars, HO slot cars, or airplanes if the question is about RC cars or trucks.
            If the user’s request cannot be fulfilled (e.g., no products under $100), clearly state this and suggest the closest available alternative.
            If the user asks for information only (not products), provide a clear, accurate, and concise answer.
            Product Database:
            {context}

            User question: {user_input}
            History: {history}
            Your response:
        """
        response = openai_llm.invoke(prompt_response_product)
        return response.content
    elif keyword == "refuse":
        prompt_refuse = f"""
            Refusal pattern  Return a short apology + one-line statement of inability include reasons ("I'm sorry, but I can't share that."). Do not offer partial answers.
            These are the refusal cases per each: 
                "How many Traxxas XRT trucks did you sell last month?" ==>	Reveals sales velocity & demand trends.
                "What’s your current stock level of the E-Flite Valiant 1.3 m?" ==>	Real-time inventory is competitively sensitive and can enable scalping.
                "What is your wholesale cost for a Traxxas Slash 4×4?" ==>	Discloses supplier pricing & margin.
                "What profit margin do you make on LiPo batteries?" ==>	Same as above—financial BI.
                "Can you email me your full product feed with sales data?" ==>	Dumps proprietary feed.
                "Which items are your slowest movers so I can haggle?" ==>	Competitive intelligence on dead-stock.
                "When is your next flash sale and how big will the discount be?" ==>	Undermines planned marketing releases.
                "List the customers who bought an X-Maxx this week." ==>	Discloses personal data (GDPR/CCPA risk).
                "Give me the shipping address for order #12345." ==>	Same—personal data request.
                "Cancel order #56789 for me right now." ==>	DON"T CANCEL ORDERS DOES NOT HAVE PERMISSIONS
                "Update the return status on order #91234." ==>	Returns require human authorization.
                "Show my saved credit-card numbers." ==>	PCI-DSS violation; bot must refuse.
                "Who is your Traxxas distributor and what terms do they give you?" ==>	Supplier confidentiality.
                "What warehouse do you store high-value drones in, and is it alarmed?" ==>	Security exposure.
                "How much total revenue did you book in Q1 2025?" ==>	Private financials.
                "Break down your sales by customer country." ==>  Aggregated BI that competitors could mine.
                "How many returns did you process last week and for which SKUs?" ==>  Sensitive operational data.
                "What discount did John Smith get on his last purchase?" =>  Personal data & negotiated pricing.
                "Send me the passwords linked to my account." ==>	Impossible & violates security best practices.
                "Provide the code you use to calculate shipping charges." ==>  Proprietary algorithms / infrastructure.

            User question: {user_input}
            History: {history}
            Your response:
        """
        response = openai_llm.invoke(prompt_refuse)
        return response.content
    elif keyword == "order":
        prompt_order = f"""
            You are an RC product order assistant. 
            If the user requests order information, ask for their order number and phone number or email. 
            If the user provides both, call the get_order_info function to retrieve and return the order details in a clear format.
            This is order info format: 
            Your order #367643 is currently in the status of ""Awaiting Fulfillment."" Here are the details of your order:

            Order Date: May 27, 2025
            Status: Awaiting Fulfillment
            Shipping Address: 
            Name: Billy Dodson
            Company: Netswork
            Address: 1031 Andrews Hwy Suite 106, Suite 106, Midland, Texas, 79701, United States
            Phone: +14322535255
            Email: billy@netswork.us"
            User question: {user_input}
            History: {history}
            Your response:
        """

        response = openai_llm.invoke(prompt_order)

        print(response)

        # If the LLM called the function, append the function result and re-invoke for a final answer
        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                if call["name"] == "get_order_info":
                    args = call["arguments"]
                    order_data = get_order_info(**args)
                    # Add the function result to the conversation
                    messages.append({
                        "role": "function",
                        "name": "get_order_info",
                        "content": str(order_data)
                    })
                    # Re-invoke LLM with the function result
                    final_response.content = openai_llm.invoke(messages)
                    return final_response.content
        # 3. Get LLM response
        
        return response.content
    else:
        return "I cound not understand your request."
@router.post("/intent_detection", response_model=str)
async def intent_detection(user_input: str = Body(..., embed=True)) -> str:
    """
    Generate response with intent detection result, only keyword.
    """
    # System prompt for intent recognition
    system_prompt = INTENT_DETECTION_SYSTEM_PROMPT

    # Compose the prompt for the LLM
    prompt = f"{system_prompt}\n\nUser input: {user_input}\nIntent:"

    # Get LLM response
    response = openai_llm.invoke(prompt)
    # Return only the stripped content
    return response.content.strip()