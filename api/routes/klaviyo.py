from fastapi import APIRouter, HTTPException, Query, Body
from langchain_openai import ChatOpenAI
import requests
import os
import json

router = APIRouter()

# Set these as environment variables or config
BOTPRESS_URL = os.getenv("BOTPRESS_URL", "https://api.botpress.cloud")  # e.g., "http://localhost:3000"
BOTPRESS_BOT_ID = os.getenv("BOTPRESS_BOT_ID", "")
BOTPRESS_ACCESS_TOKEN = os.getenv("BOTPRESS_ACCESS_TOKEN", "")

KLAVIYO_API_KEY = os.getenv("KLAVIYO_API_KEY", "")
KLAVIYO_PROFILE_URL = "https://a.klaviyo.com/api/profiles/"
KLAVIYO_EVENT_URL = "https://a.klaviyo.com/api/events/"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "got-4.1")

KLAVIYO_HEADERS = {
    "Authorization": f"Klaviyo-API-Key {KLAVIYO_API_KEY}",
    "accept": "application/vnd.api+json",
    "content-type": "application/vnd.api+json",
    "revision": "2025-04-15"
}

openai_llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

@router.post("/sync-botpress-conversations-to-klaviyo/")
async def sync_conversations_to_klaviyo():
    table_name = "Int_Connor_Conversations_Table"
    BOTPRESS_URL = f"https://api.botpress.cloud/v1/tables/{table_name}/rows/find"
    
    # Botpress headers
    headers = {
        "Authorization": f"bearer {BOTPRESS_ACCESS_TOKEN}",
        "x-bot-id": BOTPRESS_BOT_ID,
        "Content-Type": "application/json"
    }

    payload = {
        "limit": 100,
        "offset": 0
    }

    try:
        response = requests.post(BOTPRESS_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        rows = data.get("rows", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Botpress fetch error: {e}")

    sent_count = 0
    for row in rows:
        conversation_id = row.get("conversationId", f"noid_{row.get('id')}")
        transcript = row.get("transcript")
        prompt_order = f"""
            Only return email address using conversation transcript data, Do not inlcude anything else.
            History: {transcript}
            Your response:
        """

        response = openai_llm.invoke(prompt_order)

        print(response.content)

        email = response.content

        # 1. Create/Update Profile in Klaviyo
        profile_payload = {
            "data": {
                "type": "profile",
                "attributes": {
                    "email": email,
                    "properties": {
                        "conversationId": conversation_id,
                        "email": email,
                        "createdAt": row.get("createdAt"),
                        "updatedAt": row.get("updatedAt"),
                        "topics": row.get("topics"),
                        "sentiment": row.get("sentiment"),
                        "resolved": row.get("resolved"),
                        "escalations": row.get("escalations"),
                    }
                }
            }
        }
        try:
            profile_resp = requests.post(KLAVIYO_PROFILE_URL, json=profile_payload, headers=KLAVIYO_HEADERS)
            profile_resp.raise_for_status()
            profile_id = profile_resp.json()["data"]["id"]

            # 2. Create Event in Klaviyo
            transcript = row.get("transcript", [])
            summary = row.get("summary", "")
            transcript_str = json.dumps(transcript, ensure_ascii=False)

            event_payload = {
                "data": {
                    "type": "event",
                    "attributes": {
                        "metric": {
                            "data": {
                                "type": "metric",
                                "attributes": {
                                    "name": "Chatbot Conversation"
                                }
                            }
                        },
                        "profile": {
                            "data": {
                                "type": "profile",
                                "id": profile_id
                            }
                        },
                        "properties": {
                            "conversationId": conversation_id,
                            "email": email,
                            "createdAt": row.get("createdAt"),
                            "topics": row.get("topics"),
                            "sentiment": row.get("sentiment"),
                            "resolved": row.get("resolved"),
                            "escalations": row.get("escalations"),
                            "summary": summary,
                            "transcript": transcript_str
                        }
                    }
                }
            }
            event_resp = requests.post(KLAVIYO_EVENT_URL, json=event_payload, headers=KLAVIYO_HEADERS)
            event_resp.raise_for_status()
            sent_count += 1
        except Exception as e:
            # Optionally log or collect errors for failed rows
            continue

    return {"status": "success", "rows_synced": sent_count}


@router.post("/test-botpress-table-find/")
async def test_botpress_table_find(
    payload: dict = Body(
        default={
            "limit": 100,
            "offset": 0,
        }
    )
):
    table_name = "Int_Connor_Conversations_Table"
    BOTPRESS_URL = f"https://api.botpress.cloud/v1/tables/{table_name}/rows/find"
    BOTPRESS_ACCESS_TOKEN = os.getenv("BOTPRESS_ACCESS_TOKEN")
    BOTPRESS_BOT_ID = os.getenv("BOTPRESS_BOT_ID")

    headers = {
        "Authorization": f"bearer {BOTPRESS_ACCESS_TOKEN}",
        "x-bot-id": BOTPRESS_BOT_ID,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(BOTPRESS_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}