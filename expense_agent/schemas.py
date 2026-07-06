"""Typed contracts for every boundary in the graph. Malformed data is
rejected at intake, before any tokens are spent."""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

DecisionLiteral = Literal["approve", "deny", "escalate"]


class ExpenseRequest(BaseModel):
    request_id: str
    employee_id: str
    amount: float = Field(gt=0)
    currency: str = "EUR"
    vendor: str = Field(min_length=1)
    category: str = Field(min_length=1)  # meals, travel, software, equipment, ...
    description: str = ""
    receipt_attached: bool = False
    submitted_at: Optional[datetime] = None

    def normalized(self) -> "ExpenseRequest":
        return self.model_copy(
            update={"submitted_at": self.submitted_at or datetime.now(timezone.utc)}
        )


class LLMDecision(BaseModel):
    """Structured output contract for the reasoning step. The model must
    justify its decision by citing clause IDs it was actually shown."""

    decision: DecisionLiteral
    justification: str
    cited_clause_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class GuardrailEvent(BaseModel):
    rule_id: str
    stage: Literal["pre", "post"]
    action: Literal["deny", "escalate"]
    detail: str
