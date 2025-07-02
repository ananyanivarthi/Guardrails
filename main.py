from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging
import uuid
import traceback

from app.azure_client import get_azure_client
from app.pii_utils import mask_pii  # ✅ Updated to use spaCy + IndicNER
from app.guardrails_client import load_guardrails_config
from app.models import GuardrailsRequest, GuardrailsResponse

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="General Guardrails API",
    description="A general-purpose guardrails tool with PII masking, input/output checks, hallucination detection, and safety protections.",
    version="1.0.0"
)

# Load LLM and Guardrails config
azure_llm = get_azure_client()
rails = load_guardrails_config(azure_llm)


@app.post("/guardrails", response_model=GuardrailsResponse)
async def apply_guardrails(data: GuardrailsRequest):
    try:
        logger.debug(f"Received input: {data.user_input}")
        request_id = str(uuid.uuid4())
        metadata = {"request_id": request_id}

        # --- PII Masking ---
        masked_input = mask_pii(data.user_input)
        metadata["pii_detected"] = masked_input != data.user_input

        # --- Generate LLM Response ---
        response = await rails.generate_async(messages=[{"role": "user", "content": masked_input}])
        full_output = response.get("content", "").strip()
        final_response = full_output
        is_harmful = False

        # --- Guardrails Checks ---
        if response.get("is_jailbreak_attempt", False):
            print("Jailbreak attempt detected")
            is_harmful = True
            final_response = "I'm sorry, that request appears to be an attempt to bypass safety protocols."
            metadata["check_failed"] = "jailbreak_detection"
        elif response.get("is_harmful", False):
            print("Harmful content detected")
            is_harmful = True
            final_response = "I'm sorry, your request contains harmful or abusive content. Please rephrase."
            metadata["check_failed"] = "harmful_content"
        elif "sensitive personal information" in final_response.lower():
            print("PII detected in response")
            is_harmful = False
            final_response = "PII is detected and has been masked for your safety."
            metadata["check_failed"] = "pii_check"
        elif "hallucinated" in final_response.lower() or response.get("is_hallucinated", False):
            print("Hallucination detected in response")
            is_harmful = True
            final_response = "I'm sorry, I can't provide that information as it may not be accurate."
            metadata["check_failed"] = "hallucination_detection"
        elif response.get("is_illegal", False):
            print("Illegal content detected in response")
            is_harmful = True
            final_response = "I'm sorry, your request involves illegal or unethical activities. Please rephrase."
            metadata["check_failed"] = "illegal_content"

        return GuardrailsResponse(
            is_harmful=is_harmful,
            response=final_response,
            masked_input=masked_input if metadata["pii_detected"] else data.user_input,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
