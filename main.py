from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging
import uuid
import traceback

from app.models import PiimaskingRequest,PiimaskingResponse
from app.pii_utils import mask_pii  

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="General Guardrails API",
    description="API to detect and mask Personally Identifiable Information (PII) in text..",
    version="1.0.0"
)



@app.post("/pii_masking", response_model=PiimaskingResponse)
async def apply_pii_masking(data: PiimaskingRequest):
    try:
        request_id = str(uuid.uuid4())
        user_input = data.user_input.strip()
        logger.info(f"[{request_id}] Received input: {user_input}")

        # --- PII Masking Logic ---
        masked_input = mask_pii(user_input)
        pii_detected = masked_input != user_input

        # --- Logging ---
        if pii_detected:
            logger.info(f"[{request_id}] PII detected. Masked input: {masked_input}")
        else:
            logger.info(f"[{request_id}] No PII detected.")

        if user_input != masked_input:
            logger.debug(f"[{request_id}] Diff:\n- Original: {user_input}\n- Masked:   {masked_input}")

        # --- Metadata Block ---
        metadata = {
            "request_id": request_id,
            "pii_detected": pii_detected
        }

        return PiimaskingResponse(
            masked_input=masked_input if pii_detected else "NULL",
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
