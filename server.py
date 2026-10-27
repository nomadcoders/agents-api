"""Supplied local FastAPI server. Start manually: uv run uvicorn server:app --port 8000

Each route calls one SDK method from the notebook lessons. OpenAI stores the sessions
and runs the agent; this server only exposes them over HTTP and runs our function.
"""

import base64
import json
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from openai import OpenAI, omit
from pydantic import BaseModel

load_dotenv()

client = OpenAI()
app = FastAPI(title="AX School Agents API")
model = "gpt-6-luna"

# Our function: a schema for the model and a handler for us (same as video 6)
customers = {"abc": {"name": "Alex", "plan": "Premium"}}


def get_customer(customer_id):
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id!r} was not found")
    return customers[customer_id]


# Function worker: runs in the background, so it works whether or not anyone is subscribed.
# It checks the session once a second, runs any pending function call, and stops when the
# session is idle with nothing pending.
def run_functions(session_id):
    while True:
        session = client.beta.agents.sessions.retrieve(session_id)
        for action in session.required_actions:
            if action.type != "function_call":
                continue
            call = action.to_dict()
            result = {
                "type": "agent.session.input.tool_result",
                "turn_id": call["turn_id"],
                "call_id": call["call_id"],
            }
            try:
                customer = get_customer(
                    call["arguments"]["customer_id"]
                )  # pyright: ignore[reportIndexIssue]
                result["output"] = json.dumps(customer)
                result["success"] = True
            except ValueError as error:
                result["error"] = str(error)
                result["success"] = False
            client.beta.agents.sessions.events.create(
                session_id,
                events=[result],  # pyright: ignore[reportArgumentType]
            )
        if session.status == "idle" and not session.required_actions:
            return
        time.sleep(1)


# Sessions
class NewSession(BaseModel):
    message: str


@app.post("/sessions", status_code=201)
def create_session(body: NewSession, background: BackgroundTasks):
    session = client.beta.agents.sessions.create(
        agent={
            "model": model,
            "tools": [
                {
                    "type": "function",
                    "name": "get_customer",
                    "description": "Look up a customer by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
        environment={"type": "openai_hosted", "network": {"access": "disabled"}},
        input=body.message,
    )
    background.add_task(run_functions, session.id)
    return session.to_dict()


@app.get("/sessions")
def list_sessions(limit: int = 20, after: str | None = None):
    return client.beta.agents.sessions.list(limit=limit, after=after or omit).to_dict()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return client.beta.agents.sessions.retrieve(session_id).to_dict()


class Update(BaseModel):
    effort: Literal["low", "medium", "high"]


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, body: Update):
    agent = client.beta.agents.sessions.update(
        session_id, agent={"reasoning": {"effort": body.effort}}
    )
    return agent.to_dict()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    return client.beta.agents.sessions.delete(session_id).to_dict()


@app.get("/sessions/{session_id}/items")
def list_items(session_id: str, limit: int = 100, after: str | None = None):
    items = client.beta.agents.sessions.items.list(
        session_id, limit=limit, after=after or omit, order="asc"
    )
    return items.to_dict()


# Input: messages, steering, and cancellation
class Message(BaseModel):
    text: str


@app.post("/sessions/{session_id}/messages", status_code=202)
def send_message(
    session_id: str,
    body: Message,
    background: BackgroundTasks,
    idempotency_key: str | None = Header(None),
):
    client.beta.agents.sessions.events.create(
        session_id,
        idempotency_key=idempotency_key or str(uuid4()),
        events=[
            {
                "type": "agent.session.input.message",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": body.text}],
                    }
                ],
            }
        ],
    )
    background.add_task(run_functions, session_id)
    return {"accepted": True}


@app.post("/sessions/{session_id}/cancel", status_code=202)
def cancel_turn(session_id: str):
    client.beta.agents.sessions.events.create(
        session_id, events=[{"type": "agent.session.input.cancel"}]
    )
    return {"accepted": True}


# Events: forward the platform stream as server-sent events
@app.get("/sessions/{session_id}/events")
def stream_events(session_id: str):
    def frames():
        with client.beta.agents.sessions.events.stream(session_id) as events:
            yield "event: server.ready\ndata: {}\n\n"  # upstream is open; safe to send now
            for event in events:
                yield f"data: {event.to_json(indent=None)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")


# Files: upload into the hosted workspace, download published artifacts
class Upload(BaseModel):
    path: str
    text: str


@app.post("/sessions/{session_id}/files", status_code=201)
def upload_file(session_id: str, body: Upload):
    environment = client.beta.agents.sessions.retrieve(session_id).environment
    data = base64.b64encode(body.text.encode()).decode()
    result = client.beta.agents.environments.files.create(
        environment.id,  # pyright: ignore[reportAttributeAccessIssue]
        type="inline",
        path=body.path,
        data=data,
    )
    return result.to_dict()


@app.get("/sessions/{session_id}/artifacts")
def list_artifacts(session_id: str):
    return client.beta.agents.sessions.artifacts.list(session_id).to_dict()


@app.get("/sessions/{session_id}/artifacts/{artifact_id}")
def download_artifact(session_id: str, artifact_id: str):
    content = client.beta.agents.sessions.artifacts.content(
        artifact_id, session_id=session_id
    )
    return Response(content.read(), media_type="application/octet-stream")
