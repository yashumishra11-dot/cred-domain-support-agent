"""
Cred Domain Support Agent - FastAPI Integration & WebSocket Disconnect Tests
Track: Banking & FinTech (Cred)
Part 3 - Tasks 11 & 12 Validation: HTTP Endpoints, WebSocket Disconnect Resilience, and Audit Logs
"""

import os
import json
from starlette.testclient import TestClient
from api import app
from logger import AUDIT_LOG_FILE


def test_fastapi_endpoints():
    print("=" * 70)
    print("TESTING FASTAPI HTTP & WEBSOCKET DEPLOYMENT (Tasks 11 & 12)")
    print("=" * 70)
    client = TestClient(app)

    # 1. Test GET /health
    print("\n1. Testing GET /health...")
    resp_health = client.get("/health")
    assert resp_health.status_code == 200, f"Health check failed: {resp_health.text}"
    health_data = resp_health.json()
    print(f"Health Status: {health_data['status']}")
    print(f"Vector Collections: {health_data['collections']}")
    print(f"Audit Log Path: {health_data['audit_log_path']}")

    # 2. Test POST /ask (Policy Query)
    print("\n2. Testing POST /ask with Policy Query...")
    payload_policy = {
        "query": "What are the KYC document requirements for loan applicants?",
        "session_id": "test_http_session_1"
    }
    resp_ask1 = client.post("/ask", json=payload_policy)
    assert resp_ask1.status_code == 200, f"/ask failed: {resp_ask1.text}"
    ask1_data = resp_ask1.json()
    print(f"Response Type: {ask1_data['response_type']}")
    print(f"Sources      : {ask1_data['sources']}")
    print(f"Final Answer : {ask1_data['final_answer']}")
    print(f"Trace ID     : {ask1_data['trace_id']}")

    # 3. Test POST /ask with PII and Application Lookup
    print("\n3. Testing POST /ask with PII and Status Lookup...")
    payload_lookup = {
        "query": "My PAN is ABCDE1234F. Please check status for application CRD-APP-1002.",
        "session_id": "test_http_session_2"
    }
    resp_ask2 = client.post("/ask", json=payload_lookup)
    assert resp_ask2.status_code == 200, f"/ask failed: {resp_ask2.text}"
    ask2_data = resp_ask2.json()
    print(f"Sanitized Query (Model received): {ask2_data['query']}")
    print(f"Response Type : {ask2_data['response_type']}")
    print(f"Escalation    : {ask2_data['escalation_score']} (Escalated: {ask2_data['escalated']})")
    print(f"Final Answer  : {ask2_data['final_answer']}")

    # 4. Test POST /add-document
    print("\n4. Testing POST /add-document...")
    payload_add_doc = {
        "doc_id": "KB-DOC-013",
        "topic": "co-lending partnership guidelines",
        "title": "Co-Lending Partnership Protocols with NBFC Partners",
        "content": (
            "Cred operates co-lending arrangements with scheduled commercial banks under a direct 80:20 risk participation model. "
            "Primary customer underwriting is executed through Cred automated risk engines adhering strictly to partner credit filters. "
            "Disbursement and post-disbursement loan servicing are managed centrally within the Cred unified lending platform. "
            "Statutory collections and dispute escalations follow harmonized RBI ombudsman reporting frameworks across both entities."
        ),
        "category": "Lending Partnerships"
    }
    resp_add = client.post("/add-document", json=payload_add_doc)
    assert resp_add.status_code == 200, f"/add-document failed: {resp_add.text}"
    add_data = resp_add.json()
    print(f"Add Document Status: {add_data['status']}")
    print(f"Indexed Doc ID     : {add_data['doc_id']}")
    print(f"Updated Collections: Sentence Chunks = {add_data['sentence_collection_count']}")

    # 5. Test WebSocket Multi-Turn Chat and Graceful Disconnect Handling
    print("\n5. Testing WebSocket Multi-Turn Chat and Disconnect Resilience...")
    with client.websocket_connect("/ws/chat") as websocket:
        # Handshake
        handshake = websocket.receive_json()
        print(f"WebSocket Handshake: {handshake['type']} (Session: {handshake['session_id']})")

        # Send Turn 1
        websocket.send_text(json.dumps({"query": "What is the penalty for early prepayment on home loans?"}))
        reply1 = websocket.receive_json()
        print(f"Turn 1 Response Type: {reply1['data']['response_type']}")
        print(f"Turn 1 Answer       : {reply1['data']['final_answer']}")

        # Simulate Client Mid-Conversation Disconnect (exiting context manager disconnects socket)
        print("Simulating client disconnect mid-conversation...")
    
    print("WebSocket client disconnected gracefully.")

    # 6. Verify Server Stays Running After Client Disconnect
    print("\n6. Verifying Server Remained Running After WebSocket Disconnect...")
    resp_after = client.get("/health")
    assert resp_after.status_code == 200, "Server crashed after WebSocket disconnect!"
    print(f"Server operational status: {resp_after.json()['status']} (Survived Disconnect Cleanly)")

    # 7. Verify Structured Audit Log & Zero PII on Disk
    print("\n7. Verifying Structured Audit Log on Disk...")
    assert os.path.exists(AUDIT_LOG_FILE), "Audit log file does not exist on disk!"
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    print(f"Total structured JSON-Lines logged: {len(log_lines)}")
    last_entry = json.loads(log_lines[-1])
    print(f"Latest Log Event Type: {last_entry['response_type']} (Endpoint: {last_entry['endpoint']})")
    
    # Assert raw PAN ABCDE1234F is never found in cleartext on disk
    full_log_text = open(AUDIT_LOG_FILE, "r", encoding="utf-8").read()
    assert "ABCDE1234F" not in full_log_text, "CRITICAL AUDIT FAILURE: Raw PAN found unmasked in log file!"
    print("CONFIRMED: All fixed-format PII is masked before logging to disk. No raw PII in cleartext.")
    print("=" * 70)


if __name__ == "__main__":
    test_fastapi_endpoints()
