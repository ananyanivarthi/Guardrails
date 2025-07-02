from pydantic import BaseModel
from typing import Optional, Dict, Any

class PiimaskingRequest(BaseModel):
    user_input: str

class PiimaskingResponse(BaseModel):
    masked_input: str
    metadata: Dict[str, Any]