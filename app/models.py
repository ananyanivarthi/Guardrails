from pydantic import BaseModel
from typing import Optional, Dict, Any

class GuardrailsRequest(BaseModel):
    user_input: str

class GuardrailsResponse(BaseModel):
    is_harmful: bool
    response: str
    masked_input: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None