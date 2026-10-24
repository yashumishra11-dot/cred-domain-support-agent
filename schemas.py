"""
Cred Domain Support Agent - Pydantic Schemas
Track: Banking & FinTech (Cred)
Part 2 - Task 9: Structured Output Schema
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentResponseSchema(BaseModel):
    """
    Standardized structured response model for the Cred Domain Support Agent.
    Every response produced by the crew and API must conform to this schema.
    """
    query: str = Field(
        ...,
        description="The customer's input query after input-side guardrail processing."
    )
    response_type: str = Field(
        ...,
        description="Type of resolution: 'POLICY_ANSWER', 'STATUS_LOOKUP', 'REFUSAL_UNGROUNDED', 'SAFETY_REJECTED', or 'ESCALATED'."
    )
    final_answer: str = Field(
        ...,
        description="The grounded, policy-compliant response presented to the user."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of knowledge base document IDs or application record IDs used as sources."
    )
    escalation_score: Optional[float] = Field(
        default=None,
        description="Calculated escalation score in [0, 1] if an application lookup was performed."
    )
    escalated: bool = Field(
        default=False,
        description="Boolean indicating whether the ticket has been escalated for senior human review."
    )
    grounded: bool = Field(
        default=True,
        description="Flag indicating whether the output passed output-side groundedness verification."
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Unique request trace ID for distributed observability and auditing."
    )


class AutogenVerdictModel(BaseModel):
    """
    Structured Pydantic model returned by the Autogen Review Stage (Task 14).
    """
    approved: bool = Field(
        ...,
        description="Whether the draft answer was approved without compliance revisions."
    )
    final_answer: str = Field(
        ...,
        description="The finalized answer (either approved verbatim or revised for policy compliance)."
    )
    reason: str = Field(
        ...,
        description="Auditing explanation of why the draft was approved or what specific revisions were applied."
    )


class AskRequest(BaseModel):
    """HTTP POST /ask request model."""
    query: str = Field(..., min_length=1, max_length=2000, description="Customer inquiry or status request.")
    session_id: Optional[str] = Field(default="default_session", description="Unique conversation session ID for multi-turn memory.")


class AddDocumentRequest(BaseModel):
    """HTTP POST /add-document request model."""
    doc_id: str = Field(..., description="Unique document ID (e.g. KB-DOC-013)")
    topic: str = Field(..., description="Topic label")
    title: str = Field(..., description="Document title")
    content: str = Field(..., min_length=20, description="Document body (2-5 sentences)")
    category: Optional[str] = Field(default="Policy Extension", description="Category tag")
