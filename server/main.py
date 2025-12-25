from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, RateLimitError
from dotenv import load_dotenv
import os
import json
import uuid
from typing import List, Dict, Any, Optional
import redis  # For Upstash

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

redis_url = os.getenv("REDIS_URL")
redis_token = os.getenv("REDIS_TOKEN")  # If separate; else in URL

# Redis client with fallback
try:
    if redis_token:
        r = redis.from_url(redis_url, password=redis_token, decode_responses=True)
    else:
        r = redis.from_url(redis_url, decode_responses=True)
    r.ping()  # Test connect
    print("Redis connected (Upstash)")
except Exception as e:
    print(f"Redis connect failed ({e}), using in-memory fallback")
    r = None  # Fallback to in-memory below

# In-memory fallback if Redis fails
memory_store = {} if r is None else None

app = FastAPI(title="Groq AI Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=api_key)

# Fixed fallback sequence (start with Llama 70B, then others)
MODEL_SEQUENCE = [
    "llama-3.3-70b-versatile",  # Primary: Fast/reliable
    "mixtral-8x7b-32768",       # Next: Reasoning
    "llama-3.1-8b-instant",     # Fast fallback
    "llama-3.2-11b-vision-preview",  # Vision (handles text too)
]

VISION_MODELS = {"llama-3.2-11b-vision-preview"}

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]  # Incoming user msg only
    session_id: str  # From frontend

class ChatResponse(BaseModel):
    response: str
    used_model: str
    session_id: str

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Load messages and current_idx from Redis or in-memory"""
    if r:
        data = r.get(f"session:{session_id}")
        return json.loads(data) if data else {"messages": [], "current_model_idx": 0}
    else:
        return memory_store.get(session_id, {"messages": [], "current_model_idx": 0})

def save_session_data(session_id: str, data: Dict[str, Any]):
    """Save to Redis or in-memory"""
    if r:
        r.set(f"session:{session_id}", json.dumps(data))
    else:
        memory_store[session_id] = data

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):  # Sync def, no await
    try:
        # Load session
        session_data = get_session_data(request.session_id)
        messages = session_data["messages"]
        current_idx = session_data["current_model_idx"]

        # Append incoming user message to history
        messages.append({"role": "user", "content": request.messages[0]["content"]})  # Assume single user msg

        # Detect vision
        last_content = messages[-1]["content"]
        has_image = (
            isinstance(last_content, list)
            and any(c.get("type") == "image_url" for c in last_content if isinstance(c, dict))
        )

        # Try models from current_idx onward (any error -> next)
        used_model = None
        for idx in range(current_idx, len(MODEL_SEQUENCE)):
            model = MODEL_SEQUENCE[idx]
            try:
                # Filter to vision-only if needed (but sequence has mixed)
                if has_image and model not in VISION_MODELS:
                    continue  # Skip non-vision for image req

                completion = client.chat.completions.create(  # No await
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=False,
                )
                response_text = completion.choices[0].message.content
                used_model = model

                # Success: Append assistant msg, save (reset idx? or keep for next fallback)
                messages.append({"role": "assistant", "content": response_text})
                session_data["messages"] = messages
                session_data["current_model_idx"] = 0  # Reset to primary on success
                save_session_data(request.session_id, session_data)
                break

            except (RateLimitError, Exception) as e:  # Any error -> fallback
                print(f"Error with {model}: {e}, trying next...")
                continue

        if not used_model:
            raise HTTPException(
                status_code=500,
                detail="All models failed (limits/errors). Try later."
            )

        return ChatResponse(response=response_text, used_model=used_model, session_id=request.session_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Groq Backend Ready (seamless fallback + Redis memory)"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)