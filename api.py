"""
Cred Domain Support Agent - FastAPI Backend Deployment
Track: Banking & FinTech (Cred)
Part 3 - Tasks 11 & 12: HTTP Endpoints (POST /ask, POST /add-document),
                        WebSocket Multi-Turn Chat with Disconnect Resilience,
                        and PII-Masked Structured Audit Logging.
"""

import os
import time
import uuid
import json
import asyncio
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import AgentResponseSchema, AskRequest, AddDocumentRequest
from crew import execute_support_query, get_session_history
from knowledge_base import KNOWLEDGE_BASE_DOCUMENTS
from rag_core import get_rag_manager
from logger import GLOBAL_AUDIT_LOGGER
from governance import GLOBAL_RESPONSE_CACHE

app = FastAPI(
    title="Cred Domain Support Agent API",
    description="Production-grade domain support agent for Cred lending operations and policy guidance.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def serve_chat_ui():
    """Serves the interactive Cred Domain Support Agent Web Interface."""
    if os.path.exists(STATIC_INDEX_FILE):
        with open(STATIC_INDEX_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Cred Support Agent API</h2><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")


@app.get("/health")
def health_check():
    """Health and operational telemetry check."""
    rag = get_rag_manager()
    return {
        "status": "healthy",
        "collections": {
            "fixed_chunk_count": rag.coll_fixed.count(),
            "sentence_chunk_count": rag.coll_sentence.count()
        },
        "cache_stats": GLOBAL_RESPONSE_CACHE.get_stats(),
        "audit_log_path": GLOBAL_AUDIT_LOGGER.log_path
    }


@app.post("/ask", response_model=AgentResponseSchema)
def ask_support_agent(request: AskRequest):
    """
    HTTP POST /ask
    Accepts user inquiry or loan application lookup query.
    Processes query through input guardrails, memory, CrewAI crew, and output guardrails.
    Returns structured AgentResponseSchema and writes ELK-style structured audit log.
    """
    t0 = time.perf_counter()
    trace_id = f"trc-{uuid.uuid4().hex[:12]}"
    session_id = request.session_id or "default_session"

    try:
        response = execute_support_query(
            query=request.query,
            session_id=session_id,
            trace_id=trace_id
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Log structured audit event with PII masking guaranteed
        GLOBAL_AUDIT_LOGGER.log_event(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            raw_query=request.query,
            session_id=session_id,
            latency_ms=latency_ms,
            status_code=200,
            response_type=response.response_type,
            response_data=response.model_dump(),
            error=None
        )

        return response

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        GLOBAL_AUDIT_LOGGER.log_event(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            raw_query=request.query,
            session_id=session_id,
            latency_ms=latency_ms,
            status_code=500,
            response_type="ERROR",
            response_data=None,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document")
def add_knowledge_base_document(req: AddDocumentRequest):
    """
    HTTP POST /add-document
    Adds a new verified policy document to the Cred Knowledge Base and indexes it
    into both ChromaDB collections immediately.
    """
    t0 = time.perf_counter()
    trace_id = f"trc-{uuid.uuid4().hex[:12]}"

    new_doc = {
        "doc_id": req.doc_id,
        "topic": req.topic,
        "title": req.title,
        "content": req.content,
        "metadata": {"category": req.category or "Policy Extension", "version": "1.0"}
    }

    # Add to in-memory store
    KNOWLEDGE_BASE_DOCUMENTS.append(new_doc)

    # Index into ChromaDB collections
    rag = get_rag_manager()
    rag.index_documents([new_doc])

    latency_ms = (time.perf_counter() - t0) * 1000

    GLOBAL_AUDIT_LOGGER.log_event(
        trace_id=trace_id,
        endpoint="/add-document",
        method="POST",
        raw_query=f"Added doc {req.doc_id}: {req.title}",
        session_id="admin_session",
        latency_ms=latency_ms,
        status_code=201,
        response_type="DOCUMENT_INDEXED",
        response_data={"doc_id": req.doc_id, "title": req.title},
        error=None
    )

    return {
        "status": "success",
        "message": f"Document '{req.doc_id}' successfully indexed in both vector collections.",
        "doc_id": req.doc_id,
        "fixed_collection_count": rag.coll_fixed.count(),
        "sentence_collection_count": rag.coll_sentence.count(),
        "trace_id": trace_id
    }


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time multi-turn chat.
    Gracefully handles client disconnection mid-conversation (catches WebSocketDisconnect)
    and maintains server stability for all other connected clients.
    """
    await websocket.accept()
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"

    try:
        # Send initial connection handshake
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "session_id": session_id,
            "message": "Connected to Cred Support Agent real-time stream."
        })

        while True:
            # Receive client message
            data = await websocket.receive_text()
            t0 = time.perf_counter()
            trace_id = f"trc-ws-{uuid.uuid4().hex[:8]}"

            # Parse input as plain text or JSON
            query = data
            custom_sid = session_id
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    query = parsed.get("query", data)
                    custom_sid = parsed.get("session_id", session_id)
            except Exception:
                pass

            # Execute pipeline off the main event loop
            response = await asyncio.to_thread(
                execute_support_query,
                query=query,
                session_id=custom_sid,
                trace_id=trace_id
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            # Log event
            GLOBAL_AUDIT_LOGGER.log_event(
                trace_id=trace_id,
                endpoint="/ws/chat",
                method="WEBSOCKET",
                raw_query=query,
                session_id=custom_sid,
                latency_ms=latency_ms,
                status_code=200,
                response_type=response.response_type,
                response_data=response.model_dump(),
                error=None
            )

            # Send structured response
            await websocket.send_json({
                "type": "AGENT_RESPONSE",
                "data": response.model_dump()
            })

    except WebSocketDisconnect:
        # Graceful disconnect handling required by Task 11:
        # Catch WebSocketDisconnect, log clean closure, and keep server running for other clients
        GLOBAL_AUDIT_LOGGER.log_event(
            trace_id=f"trc-disc-{uuid.uuid4().hex[:6]}",
            endpoint="/ws/chat",
            method="WEBSOCKET_DISCONNECT",
            raw_query="[Client Disconnected]",
            session_id=session_id,
            latency_ms=0.0,
            status_code=1000,
            response_type="CLIENT_DISCONNECTED",
            response_data=None,
            error="Client gracefully disconnected."
        )
    except Exception as e:
        GLOBAL_AUDIT_LOGGER.log_event(
            trace_id=f"trc-err-{uuid.uuid4().hex[:6]}",
            endpoint="/ws/chat",
            method="WEBSOCKET_ERROR",
            raw_query="[WebSocket Error]",
            session_id=session_id,
            latency_ms=0.0,
            status_code=1011,
            response_type="WEBSOCKET_ERROR",
            response_data=None,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Cred Domain Support Agent FastAPI server on port 8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
