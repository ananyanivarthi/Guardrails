import os
import logging
import traceback
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

def load_guardrails_config(llm: AzureChatOpenAI) -> LLMRails:
    try:
        logger.info("Loading Guardrails configuration")
        rails_config_path = os.path.join(os.getcwd(), "config")
        config = RailsConfig.from_path(rails_config_path)

        logger.info("Initializing LLMRails with NeMo config")
        rails = LLMRails(config=config, llm=llm)
        return rails

    except Exception as e:
        logger.error(f"Guardrails initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise
