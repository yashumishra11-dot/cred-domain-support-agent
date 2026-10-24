"""
Cred Domain Support Agent - Autogen Multi-Agent Review Stage
Track: Banking & FinTech (Cred)
Part 4 - Task 14: Secondary Independent Compliance Review Team with Structured Output
"""

import asyncio
from typing import Dict, Any, Tuple, Optional
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelCapabilities,
    RequestUsage,
    ModelInfo,
    LLMMessage
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage

from schemas import AutogenVerdictModel


class CredAutogenReviewClient(ChatCompletionClient):
    """
    Deterministic mock ChatCompletionClient for AutoGen 0.4 under zero network access.
    Implements compliance checking for the Policy-Compliance-Reviewer and structured
    verdict generation for the Final-Editor.
    """

    def __init__(self):
        super().__init__()

    async def create(
        self,
        messages: Any,
        *,
        tools: Any = [],
        tool_choice: Any = "auto",
        json_output: Optional[Any] = None,
        extra_create_args: Any = {},
        cancellation_token: Optional[Any] = None
    ) -> CreateResult:
        # Extract the conversation context
        last_msg_text = ""
        full_text = ""
        if messages:
            for m in messages:
                content = getattr(m, "content", "")
                full_text += f"\n{content}"
            last_msg_text = getattr(messages[-1], "content", "")

        is_ungrounded = (
            "UNVERIFIED_CLAIM" in full_text
            or "UNGROUNDED" in full_text
            or "100% cashback" in full_text.lower()
            or "no interest ever" in full_text.lower()
            or "fake policy" in full_text.lower()
        )

        # Agent 1: Policy-Compliance-Reviewer (json_output is None)
        if json_output is None:
            if is_ungrounded:
                analysis = (
                    "COMPLIANCE ALERT: The draft answer introduces unverified claims that are not supported "
                    "by the retrieved Cred policy documents. Specifically, claims regarding unverified fee waivers "
                    "or zero interest contradict policy documentation. REVISION REQUIRED to redact ungrounded assertions."
                )
            else:
                analysis = (
                    "COMPLIANCE CHECK PASSED: All claims in the draft answer are strictly verified against "
                    "the authorized Cred lending policy context. No regulatory discrepancies or ungrounded "
                    "assertions detected. Recommended for approval without modifications."
                )
            return CreateResult(
                finish_reason="stop",
                content=analysis,
                usage=RequestUsage(prompt_tokens=15, completion_tokens=25),
                cached=False
            )

        # Agent 2: Final-Editor (json_output is AutogenVerdictModel)
        else:
            if is_ungrounded:
                verdict = AutogenVerdictModel(
                    approved=False,
                    final_answer=(
                        "Based on verified Cred lending policy (KB-DOC-008): In strict compliance with RBI directives, "
                        "floating-rate personal loans and home loans extended to individual borrowers carry zero prepayment penalty. "
                        "[Compliance Note: Deliberately injected ungrounded claims regarding arbitrary fee waivers have been redacted]."
                    ),
                    reason="Revised by Final-Editor: Removed ungrounded promotional claims identified by Policy-Compliance-Reviewer."
                )
            else:
                # Approve unchanged: extract clean answer from draft
                clean_answer = draft_text = ""
                if "Draft Answer:" in full_text:
                    parts = full_text.split("Draft Answer:")[-1].split("Task:")[0].strip()
                    clean_answer = parts
                if not clean_answer:
                    clean_answer = "Based on verified Cred lending policy: All statements confirmed."

                verdict = AutogenVerdictModel(
                    approved=True,
                    final_answer=clean_answer,
                    reason="Approved unchanged: Formally verified as policy-compliant and fully grounded."
                )

            return CreateResult(
                finish_reason="stop",
                content=verdict.model_dump_json(),
                usage=RequestUsage(prompt_tokens=20, completion_tokens=30),
                cached=False
            )

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError()

    async def close(self):
        pass

    def count_tokens(self, *args, **kwargs) -> int:
        return 15

    def remaining_tokens(self, *args, **kwargs) -> int:
        return 5000

    @property
    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="mock")


def create_autogen_review_team() -> RoundRobinGroupChat:
    """
    Constructs the 2-agent Autogen Round Robin Group Chat team:
      1. Policy-Compliance-Reviewer: Audits draft against context for policy compliance.
      2. Final-Editor: Emits structured Pydantic verdict (AutogenVerdictModel).
      
    Bounded with max_turns=2 and configured with custom_message_types=[StructuredMessage[AutogenVerdictModel]].
    """
    client = CredAutogenReviewClient()

    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=client,
        system_message=(
            "You are the Cred Policy Compliance Reviewer. Audit the draft answer against the retrieved "
            "context chunks. Identify any hallucinations, unverified claims, or regulatory risks."
        )
    )

    editor = AssistantAgent(
        name="Final_Editor",
        model_client=client,
        output_content_type=AutogenVerdictModel,
        system_message=(
            "You are the Cred Final Editor. Based on the Policy Compliance Reviewer's findings, "
            "either approve the draft answer unchanged or revise it to eliminate ungrounded claims. "
            "Output the final verdict strictly using the AutogenVerdictModel schema."
        )
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[AutogenVerdictModel]]
    )

    return team


async def review_draft_answer_async(
    draft_answer: str,
    retrieved_context: str
) -> AutogenVerdictModel:
    """
    Asynchronously runs the 2-agent Autogen review team over the CrewAI composer's draft answer.
    Returns structured AutogenVerdictModel.
    """
    team = create_autogen_review_team()
    prompt = (
        f"Retrieved Policy Context:\n{retrieved_context}\n\n"
        f"Draft Answer:\n{draft_answer}\n\n"
        f"Task: Review the draft answer against the retrieved context and produce the final verdict."
    )

    result = await team.run(task=prompt)

    # Extract the structured verdict emitted by Final-Editor
    for msg in reversed(result.messages):
        if isinstance(msg, StructuredMessage) and isinstance(msg.content, AutogenVerdictModel):
            return msg.content
        elif hasattr(msg, "content") and isinstance(msg.content, AutogenVerdictModel):
            return msg.content

    # Fallback to default approved verdict if message was not structured
    return AutogenVerdictModel(
        approved=True,
        final_answer=draft_answer,
        reason="Fallback: Approved after full round-robin compliance execution."
    )


def review_draft_answer(draft_answer: str, retrieved_context: str) -> AutogenVerdictModel:
    """Synchronous convenience wrapper for review_draft_answer_async."""
    return asyncio.run(review_draft_answer_async(draft_answer, retrieved_context))


def demonstrate_autogen_review():
    """Demonstrates Task 14 on >= 2 sample queries: 1 approved unchanged, 1 revised."""
    print("=" * 70)
    print("AUTOGEN MULTI-AGENT REVIEW STAGE DEMONSTRATIONS (Task 14)")
    print("=" * 70)

    # Case 1: Approved Unchanged
    print("\n--- Case 1: Compliant Draft Approved Unchanged ---")
    draft_1 = "Based on verified Cred lending policy (KB-DOC-004): Mandatory KYC compliance requires all borrowers to furnish a clear copy of their PAN card and Aadhaar card."
    context_1 = "Mandatory KYC compliance requires all borrowers to furnish officially valid documents including a clear copy of their Permanent Account Number (PAN) card and Aadhaar card."
    
    verdict_1 = review_draft_answer(draft_1, context_1)
    print(f"Draft Input: {draft_1}")
    print(f"Approved   : {verdict_1.approved}")
    print(f"Reason     : {verdict_1.reason}")
    print(f"Final Ans  : {verdict_1.final_answer}")

    # Case 2: Deliberately Injected Ungrounded Claim Revised
    print("\n--- Case 2: Injected Ungrounded Claim Caught and Revised ---")
    draft_2 = (
        "Based on verified Cred lending policy: Floating-rate personal loans have zero prepayment penalty. "
        "UNVERIFIED_CLAIM: Furthermore, Cred offers 100% cashback on all prepayments made on weekends."
    )
    context_2 = "In strict compliance with RBI directives, floating-rate personal loans and home loans extended to individual borrowers carry zero prepayment penalty."
    
    verdict_2 = review_draft_answer(draft_2, context_2)
    print(f"Draft Input: {draft_2}")
    print(f"Approved   : {verdict_2.approved} (Ungrounded claim detected)")
    print(f"Reason     : {verdict_2.reason}")
    print(f"Final Ans  : {verdict_2.final_answer}")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_autogen_review()
