import os
import logging
import traceback
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def get_azure_client():
    try:
        logger.info("Initializing Azure Open AI client")
        azure_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-01-01-preview",
            deployment_name="gpt-4o",
            max_tokens=200,
            temperature=0.7,
            n=1,
            top_p=0.9
        )
        return azure_llm
    except Exception as e:
        logger.error(f"Azure Open AI initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise
