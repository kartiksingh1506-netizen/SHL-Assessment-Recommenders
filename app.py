from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from embedding import init, get_index, get_model, get_assessments

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0"
)

init()

# -----------------------------
# Request Models
# -----------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Chat Endpoint
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    # Get latest user message
    latest_message = ""

    for message in reversed(request.messages):
        if message.role == "user":
            latest_message = message.content
            break

    # Handle vague queries
    vague_queries = [
        "assessment", "test", "recommend",
        "hire", "hiring", "candidate", "job"
    ]

    if (
        len(latest_message.split()) < 4
        or latest_message.lower().strip() in vague_queries
    ):
        return {
            "reply": (
                "Please provide more details like job role, skills, or job description."
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # -----------------------------
    # Semantic Search
    # -----------------------------
    model = get_model()
    index = get_index()
    assessments = get_assessments()

    query_embedding = model.encode([latest_message])
    distances, indices = index.search(query_embedding, 10)

    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(assessments):
            item = dict(assessments[idx])
            item["score"] = float(dist)
            results.append(item)

    # sort best matches
    results = sorted(results, key=lambda x: x["score"])

    # build response
    recommendations = []

    for assessment in results[:5]:
        recommendations.append({
            "name": assessment.get("name"),
            "url": assessment.get("link"),
            "test_type": ", ".join(assessment.get("keys", []))
        })

    return {
        "reply": f"I found {len(recommendations)} matching SHL assessments.",
        "recommendations": recommendations,
        "end_of_conversation": True
    }