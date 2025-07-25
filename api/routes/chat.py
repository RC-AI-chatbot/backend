from fastapi import APIRouter, Body
from dotenv import load_dotenv
from pydantic import BaseModel
import random
import asyncio
import os
import re
import httpx
import json
import requests
import typesense

from fastapi.responses import StreamingResponse
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
BIGCOMMORCE_STORE_HASH_ID = os.getenv("BIGCOMMORCE_STORE_HASH_ID", "")
BIGCOMMERCE_ACCESS_TOKEN = os.getenv("BIGCOMMERCE_ACCESS_TOKEN", "")
BOTPRESS_WEBHOOK_URL="https://webhook.botpress.cloud/7e539cba-70f5-466d-aee5-71e4ff7ecfd7"


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "FastAPI makes building APIs fast and fun.",
    "Remote control cars are exciting for all ages.",
    "Python is a versatile programming language.",
    "Streaming responses can improve user experience.",
    "Testing APIs is crucial for robust applications.",
    "Random sentences make for interesting tests.",
    "Chatbots are becoming increasingly popular.",
    "Always write clean and maintainable code."
]

openai_llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=20000,
    timeout=1000,
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
#    summary: str
    conversationId: str

def replace_car_with_truck(text):
    # Replace 'car' with 'truck', case-insensitive, word-boundary
    return re.sub(r'\bcar\b', 'truck', text, flags=re.IGNORECASE)

@router.post("/chat_with_openai", response_model=str)
async def chat_with_openai(user_input: str = Body(..., embed=True)) -> str:
    """
    Generate a response using OpenAI via LangChain.
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

@router.post("/chat_with_ai")
async def chat_with_ai(request: ChatRequest):
    """
    AI chat using Typesense as retrieval backend (RAG style).
    """
    import time
    
    async def ai_stream():
        start_time = time.time()
        user_input = request.user_input
        print(user_input)
        history = request.history # + f"\nSummary: {request.summary}"
        conversationId = request.conversationId
        
        print(f"Request processing started at: {start_time}")

        detect_openai_llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            max_tokens=20000,
            timeout=1000,
            max_retries=2,
            api_key=OPENAI_API_KEY,
        )
        
        # Detect user's input as keyword
        detect_system_prompt = INTENT_DETECTION_SYSTEM_PROMPT
        detect_system_prompt = [
            ("system", INTENT_DETECTION_SYSTEM_PROMPT),
            ("human", f'User Input: {user_input} Chat History: {history}'),
        ]

        intent_start = time.time()
        keyword = detect_openai_llm.invoke(detect_system_prompt).content.strip()
        intent_end = time.time()
        
        print(f"Keyword: {keyword}")
        print(f"Intent detection took: {intent_end - intent_start:.2f} seconds")

        if keyword == "product" or keyword == "response":
            product_start = time.time()
            user_input.replace("1000", "700~1000")
            user_input.replace("500", "300~500")
            
            # Vector store setup
            vector_setup_start = time.time()
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
                search_kwargs={'score_threshold': 0.8, 'k': 4}
            )
            vector_setup_end = time.time()
            print(f"Vector store setup took: {vector_setup_end - vector_setup_start:.2f} seconds")

            SYSTEM_TEMPLATE = """
                You are an expert RC product assistant. Your job is to analyze the user’s intent and return precise, non-repetitive product results or answers, using only the provided product data.

                Strict Deduplication Rules:
                - You don't need to say greeting 
                - if user request product. just response for user's request. 
                - In this special cases, user request prodct, and ai request email and name, so user repond already, then you should response for user's last request.
                - When AI asks for a name and email, and the user answers, instead of asking them what product they are looking for, they should respond to the previous request, so in this case,  the keyword should be the product. so you should response for user's last request before providing name and email.
                - If you have responded to a product request in the previous conversation history, you do not need to ask for your name and email address anymore and you should respond to user's request.
                - Never return more than one product with the same parent SKU, title, or identical features, even if they differ only by color or minor options.
                - Do not list multiple variants of the same base product. Only include the most popular or relevant variant (based on total_sales, in_stock).
                - Do not repeat products in any response.
                - Return a maximum of 2 products (with unique parent SKUs) unless the user asks for more (never return more than 5).
                - If it's hard to determine the criteria from the user's request, it will revert back to the views and sales rank of the product.
                - If the user asks for the reason for recommending this product, it will revert back to the reasons for recommending the previously recommended products.  
                Default Sorting and Filtering:
                - By default, sort products by total_sales (highest first), in_stock = Yes.
                If a price range is given, only include products with price or sale_price within that range.
                - If a user requests skill level (beginner/intermediate/advanced): select products with matching tags or, if tags are missing, use product descriptions to infer suitability.
                - For compatibility or upgrade requests: Only show parts/upgrades specifically compatible with the mentioned model.
                - Group results by category if relevant (e.g., “Wheels”, “Motors”, “Body Shells”).
                - Even if the title is almost the same but the color at the end is different, I will consider it as the same product and return only one.

                Response Formatting:
                    For each product, provide:
                    Number and Product Title (in bold).
                    Short product description.
                    Price.
                    “Show Now” as a button with the product link.
                    Product image in markdown format, resized to 50-75%.
                        Example: ![Alt text](image_url)
                
                List only 2 products (unique parent SKU) unless more are requested (max 5).
                Do not show different color options as separate products.
                Never repeat products (by title, features, or parent SKU).
                Even if the title is almost the same but the color at the end is different, I will consider it as the same product and return only one.

                Special Cases:
                If user asks for “most popular RC cars,” respond:
                    “Here are the most popular RC cars currently available:”
                    (if 2 are color variants, state: “2 of these are variants of the same car.”)
                If user requests RC vehicles under $100 and none are available:
                    “Sorry, we currently don’t have RC cars under $100. Our least expensive is…” (then show the least expensive).
                If no products fit the user’s constraints, clearly state this and recommend    the closest available alternative.
                Even if the title is almost the same but the color at the end is different, I will consider it as the same product and return only one.
                
                General:
                - For technical, compatibility, or installation questions: provide clear, concise, and factual answers strictly based on product data.
                - If the user requests information only (not products): provide an accurate, concise answer.
                - Never list products outside the provided categories or context.
                Consider “car,” “buggy,” “truck,” etc. as equivalent for user intent.
                
                Remember:
                Your top priority is to ensure that each product shown is unique in its title and features and model, never differing only by color or minor cosmetic options. Never repeat products in the results.

                Template for use in retrieval-augmented scenarios:
                - After retrieving candidate products, remove all duplicate or near-duplicate products (same parent SKU/title/features/differ only by color).
                - Only display unique, most relevant products as described above.
                Product Database:
                <context>
                {context}
                <context>
                Your response:
            """

            # Chain setup
            chain_setup_start = time.time()
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
            chain_setup_end = time.time()
            print(f"Chain setup took: {chain_setup_end - chain_setup_start:.2f} seconds")

            # Stream generation
            stream_start = time.time()
            all_content = ""
            
            # Replace streaming with direct invoke
            response = conversational_retrieval_chain.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input),
                    ]
                },
            )

            # for chunk in stream:
            #     for key in chunk:
            #         if key == "answer":
            #             all_content += chunk[key]
            if "answer" in response: 
                all_content = response["answer"]
                
            stream_end = time.time()
            print(f"Stream generation took: {stream_end - stream_start:.2f} seconds")

            # Send to Botpress
            botpress_start = time.time()
            await send_chunk_to_botpress(all_content, conversationId)
            botpress_end = time.time()
            print(f"Botpress sending took: {botpress_end - botpress_start:.2f} seconds")
            
            product_end = time.time()
            print(f"Total product/response handling took: {product_end - product_start:.2f} seconds")
            
            yield all_content
        elif keyword == "refuse":
            refuse_start = time.time()
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
            buffer = ""
            llm_start = time.time()
            for chunk in openai_llm.stream(prompt_refuse):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            llm_end = time.time()
            print(f"LLM streaming for refuse took: {llm_end - llm_start:.2f} seconds")
            
            botpress_start = time.time()
            await send_chunk_to_botpress(buffer, conversationId)
            botpress_end = time.time()
            print(f"Botpress sending took: {botpress_end - botpress_start:.2f} seconds")
            
            refuse_end = time.time()
            print(f"Total refuse handling took: {refuse_end - refuse_start:.2f} seconds")
        elif keyword == "order":
            order_start = time.time()
            prompt_order = f"""
                If the user does not enter at least one of the phone number or email address, then ask them to enter at least one of them for security check.
                If user provide order number then just return json format with order number and phone number and email address with #####, nothing else:
                else If the user requests order information, ask for their order number and phone number or email. 
                
                else If the user not provide order number, ask again order number
                
                User input: {user_input}
                ChatHistory: {history}
                
            """
            response = openai_llm.invoke(prompt_order)

            response_content = response.content

            print("AI response: ", response_content)
            try:
                # Try to parse as JSON
                parsed = json.loads(response_content)

                print("AI parse result: ",parsed)
                order_number = parsed.get("order_number")
                phone_number = parsed.get("phone_number")
                email_address = parsed.get("email_address")
                print("AI parse result: ",order_number)
                # Now you can use these variables as needed
                # Example: return or store them
                order_info = get_bigcommerce_order(order_number)
                order_shipping_addresses = get_bigcommerce_order_shipping_addresses(order_number)
                print(order_info)

                prompt_order = f"""
                    Based on this order info : {order_info} and order shipping addresses : {order_shipping_addresses}
                    Display order detail info like this format: 

                    1. Do not show order Billing Address
                    2. Show shipping address Only
                    
                    you can get those detail infos from provided order_info and order shipping addresses.

                    Before return order detail info, we should check the security with email address or phone number at least one.
                    email address is {email_address} and phone number is {phone_number}
                    If email address or phone number is not match with user's providing, Do not return order detail and return "Sorry, your providing order information is not match with order id. please input correct email address/phone number"
                    else if match, Return the order details.

                    FORMAT LIKE THIS
                        Your order #367643 is currently in the status of "Awaiting Fulfillment." Here are the details of your order:

                        Order Date: May 27, 2025
                        Status: Awaiting Fulfillment
                        Shipping Address: (find shipping address from order_shipping_addresses)
                        Name: (find first and last name from order_shipping_addresses)
                        Company: Netswork
                        Address: 1031 Andrews Hwy Suite 106, Suite 106, Midland, Texas, 79701, United States
                        Phone: +14322535255
                        Email: (find email address from order_shipping_addresses)
                    
                    If order_info is not correct, just return "Sorry, we are unable to obtain the information for that order number. Please request a different order number or ask a different question."

                    User input: {user_input}
                    ChatHistory: {history}
                """
                buffer = ""
                for chunk in openai_llm.stream(prompt_order):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    buffer += text
                    yield text
                await send_chunk_to_botpress(buffer, conversationId)
            except json.JSONDecodeError:
                await send_chunk_to_botpress(response_content, conversationId)
                yield response_content
        elif keyword == "greeting":
            greeting_start = time.time()
            prompt_greeting = f"""
                You are RC ruddy Assistant from RCsuperstore, say greeting words and ask user name or email address kindly. 
                In conversation history, if user didn't ask user name or email address. ask to user again kindly.
                User response: {user_input}
                History: {history}  
                Your response:
            """
            buffer = ""
            for chunk in openai_llm.stream(prompt_greeting):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            await send_chunk_to_botpress(buffer, conversationId)
            greeting_end = time.time()
            print(f"Total greeting handling took: {greeting_end - greeting_start:.2f} seconds")
        elif keyword == "feedback":
            feedback_start = time.time()
            prompt_feedback = f"""
                You are RC ruddy Assistant from RCsuperstore, user gave you feedback for this platform or chatbot and say to user thank you for your feedback. and keep going.
                User feedback: {user_input}
                History: {history}
                Your response:
            """
            buffer = ""
            for chunk in openai_llm.stream(prompt_feedback):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            await send_chunk_to_botpress(buffer, conversationId)
            feedback_end = time.time()
            print(f"Total feedback handling took: {feedback_end - feedback_start:.2f} seconds")
        else:
            yield "I cound not understand your request."
    return StreamingResponse(ai_stream(), media_type="text/plain")

def get_bigcommerce_order(order_id):
    store_hash = BIGCOMMORCE_STORE_HASH_ID
    access_token = BIGCOMMERCE_ACCESS_TOKEN
    if not store_hash or not access_token:
        raise Exception("Missing BigCommerce store hash or access token in environment variables.")

    try:
        url = f"https://api.bigcommerce.com/stores/{store_hash}/v2/orders/{order_id}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Auth-Token": access_token
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return "Sorry, I can't get order info."

def get_bigcommerce_order_shipping_addresses(order_id):
    store_hash = BIGCOMMORCE_STORE_HASH_ID
    access_token = BIGCOMMERCE_ACCESS_TOKEN
    if not store_hash or not access_token:
        raise Exception("Missing BigCommerce store hash or access token in environment variables.")

    try:
        url = f"https://api.bigcommerce.com/stores/{store_hash}/v2/orders/{order_id}/shipping_addresses"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Auth-Token": access_token
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return "Sorry, I can't get order info."

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

@router.get("/test_stream", response_class=StreamingResponse)
async def test_stream(conversationId: str):
    """
    Test API that streams 5 random sentences as a response,
    and sends each chunk to the Botpress webhook.
    """
    async def sentence_generator():
        for _ in range(5):
            sentence = random.choice(SENTENCES)
            await send_chunk_to_botpress(sentence, conversationId)  # Send to Botpress
            yield sentence 
            

    return StreamingResponse(sentence_generator(), media_type="text/plain")

async def send_chunk_to_botpress(chunk: str, conversationId: str):
    async with httpx.AsyncClient() as client:
        payload = {
            "message": chunk,
            "conversationId": conversationId,
        }
        headers = {
            "Content-Type": "application/json"
        }
        await client.post(BOTPRESS_WEBHOOK_URL, json=payload, headers=headers)

@router.post("/test_chat_with_ai")
async def chat_with_ai(request: ChatRequest):
    """
    AI chat using Typesense as retrieval backend (RAG style).
    """
    async def ai_stream():
        user_input = request.user_input
        print(user_input)
        history = request.history
        conversationId = request.conversationId
        # Detect user's input as keyword
        detect_system_prompt = INTENT_DETECTION_SYSTEM_PROMPT

        detect_system_prompt = [
            (
                "system",
                INTENT_DETECTION_SYSTEM_PROMPT,
            ),
            ("human", f'User Input: {user_input} Chat History: {history}'),
        ]

        # Get LLM response as keyword
        # buffer = ""
        # for chunk in openai_llm.invoke(detect_system_prompt):
        #     text = chunk.content if hasattr(chunk, "content") else str(chunk)
        #     buffer += text

        keyword = openai_llm.invoke(detect_system_prompt).content.strip()
        
        print("Keyword: ", keyword)

        if keyword == "product" or keyword == "response":
            user_input.replace("1000", "700~1000")
            user_input.replace("500", "300~500")
            # 2. Search Typesense for relevant products
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
                search_kwargs={'score_threshold': 0.8, 'k': 10}
            )

            SYSTEM_TEMPLATE = """
                You are a highly skilled RC product assistant with expertise in remote control vehicles and parts. Analyze the user’s question and generate a clear, concise, and helpful response. Your answer should directly address the user’s intent, using only the provided product data.

                Instructions:

                If the question requests product recommendations or lists, select the most relevant products using the following criteria (unless otherwise specified by the user):
                    Default sort: total_sales (highest first), and In Stock = Yes.
                    We don't need to show recommend reason. 
                    If price constraints are given: Only include products where (price or sale_price) is within the specified range.
                    
                    If compatibility or upgrade requests: Only show parts or upgrades specifically compatible with the mentioned model.
                    If user requests beginner/intermediate/advanced: Select products with a corresponding skill level tag, or use product descriptions to infer suitability.

                Product listing rules:
                
                    Show per each product like this format  
                        Number number and title with bold text.
                        Display description,
                        Display price
                        Display Show Now as button with product link (Do not display text or number on the image)
                        Display product image like this format: : ![Alt text[(image_url) to display html in markdown format

                    Show only 3 products (unique parent_sku) unless the user asks to see more. Never show more than 5 products.
                    Always format product results in a user-friendly list, including: Product Name, Short Description (if available), Price, and a short note on why it’s recommended for the query.
                    I think you are using markdown format, so using this format: ![Alt text[(image_url) such as markdown format,size of the image by 50% to 75%. return image.
                    If relevant, group products by category (e.g., “Wheels”, “Motors”, “Body Shells”) before listing.
                If you cannot find relevant products, explain why and provide helpful suggestions or alternatives. specially, if you don't have any products under any prices, return like this format : "Sorry, we currently don't have RC cars under $100, our lease expensive is .... (show the least expensive)"
                For technical, compatibility, or installation questions, provide a concise, factual answer based only on the data. If you can list compatible products or parts, do so.
                Do not list products outside of the given context or product categories.
                Truck, buggy etc is same meaning as car.
                If the user’s request cannot be fulfilled (e.g., no products under $100), clearly state this and suggest the closest available alternative.
                If the user asks for information only (not products), provide a clear, accurate, and concise answer.
                Special cases: 
                    - What are the most popular RC cars right now? : 1. Here are the most popular RC cars currently available: (don't say by sales and in stock status) 2. 2 of these are variants of the same car (different colors).
                    - What RC vehicles do you have under $200? : Showing parts. Use criteria: 1. Total_sales and In Stock = Yes and price < $200 OR Sale_price < $200
                    - Recommend some beginner-friendly RC cars below $100? : Change response to "Sorry, we currently don't have RC cars under $100, our lease expensive is .... (show the least expensive)"
                    - If user ask most popular RC cars, then don't say by sales and in stock status.
                Product Database:
                <context>
                {context}
                <context>
                Your response:
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
                        print(chunk[key])

            await send_chunk_to_botpress(all_content, conversationId)
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
            buffer = ""
            for chunk in openai_llm.stream(prompt_refuse):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            await send_chunk_to_botpress(buffer, conversationId)
        elif keyword == "order":
            prompt_order = f"""
                If the user does not enter at least one of the phone number or email address, then ask them to enter at least one of them for security check.
                If user provide order number then just return json format with order number and phone number and email address with #####, nothing else:
                else If the user requests order information, ask for their order number and phone number or email. 
                
                else If the user not provide order number, ask again order number
                
                User input: {user_input}
                ChatHistory: {history}
                
            """
            response = openai_llm.invoke(prompt_order)

            response_content = response.content

            print("AI response: ", response_content)
            try:
                # Try to parse as JSON
                parsed = json.loads(response_content)

                print("AI parse result: ",parsed)
                order_number = parsed.get("order_number")
                phone_number = parsed.get("phone_number")
                email_address = parsed.get("email_address")
                print("AI parse result: ",order_number)
                # Now you can use these variables as needed
                # Example: return or store them
                order_info = get_bigcommerce_order(order_number)
                order_shipping_addresses = get_bigcommerce_order_shipping_addresses(order_number)
                print(order_info)

                prompt_order = f"""
                    Based on this order info : {order_info} and order shipping addresses : {order_shipping_addresses}
                    Display order detail info like this format: 

                    1. Do not show order Billing Address
                    2. Show shipping address Only
                    
                    you can get those detail infos from provided order_info and order shipping addresses.

                    Before return order detail info, we should check the security with email address or phone number at least one.
                    email address is {email_address} and phone number is {phone_number}
                    If email address or phone number is not match with user's providing, Do not return order detail and return "Sorry, your providing order information is not match with order id. please input correct email address/phone number"
                    else if match, Return the order details.

                    FORMAT LIKE THIS
                        Your order #367643 is currently in the status of "Awaiting Fulfillment." Here are the details of your order:

                        Order Date: May 27, 2025
                        Status: Awaiting Fulfillment
                        Shipping Address: (find shipping address from order_shipping_addresses)
                        Name: (find first and last name from order_shipping_addresses)
                        Company: Netswork
                        Address: 1031 Andrews Hwy Suite 106, Suite 106, Midland, Texas, 79701, United States
                        Phone: +14322535255
                        Email: (find email address from order_shipping_addresses)
                    
                    If order_info is not correct, just return "Sorry, we are unable to obtain the information for that order number. Please request a different order number or ask a different question."

                    User input: {user_input}
                    ChatHistory: {history}
                """
                buffer = ""
                for chunk in openai_llm.stream(prompt_order):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    buffer += text
                    yield text
                await send_chunk_to_botpress(buffer, conversationId)
            except json.JSONDecodeError:
                await send_chunk_to_botpress(response_content, conversationId)
                yield response_content
        elif keyword == "greeting":
            prompt_greeting = f"""
                You are RC ruddy Assistant from RCsuperstore, say greeting words and ask user name or email address kindly. 
                In conversation history, if user didn't ask user name or email address. ask to user again kindly.
                User response: {user_input}
                History: {history}  
                Your response:
            """
            buffer = ""
            for chunk in openai_llm.stream(prompt_greeting):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            await send_chunk_to_botpress(buffer, conversationId)
        elif keyword == "feedback":
            prompt_feedback = f"""
                You are RC ruddy Assistant from RCsuperstore, user gave you feedback for this platform or chatbot and say to user thank you for your feedback. and keep going.
                User feedback: {user_input}
                History: {history}
                Your response:
            """
            buffer = ""
            for chunk in openai_llm.stream(prompt_feedback):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += text
                yield text
            await send_chunk_to_botpress(buffer, conversationId)
        else:
            yield "I cound not understand your request."
    return StreamingResponse(ai_stream(), media_type="text/plain")
