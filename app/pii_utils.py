import logging
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
import phonenumbers

logger = logging.getLogger(__name__)

# Load spaCy English NER model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy en_core_web_lg model successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    raise

# Load IndicNER
try:
    indic_model_name = "ai4bharat/IndicNER"
    indic_tokenizer = AutoTokenizer.from_pretrained(indic_model_name)
    indic_model = AutoModelForTokenClassification.from_pretrained(indic_model_name)
    indic_pipeline = pipeline(
        "ner",
        model=indic_model,
        tokenizer=indic_tokenizer,
        aggregation_strategy="simple"
    )
    logger.info("Loaded IndicNER model successfully.")
except Exception as e:
    logger.error(f"Failed to load IndicNER model: {e}")
    raise


def get_spacy_entities(text: str):
    doc = nlp(text)
    return [{"entity_group": ent.label_, "word": ent.text, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG"]]


def get_indic_entities(text: str):
    results = indic_pipeline(text)
    logger.info(f"Raw IndicNER results: {results}")

    merged = []
    for ent in results:
        logger.info(f"Processing token: {ent}")
        merged.append({
            "entity_group": normalize_label(ent["entity_group"]),
            "word": ent["word"],
            "start": ent["start"],
            "end": ent["end"]
        })
    logger.info(f"Merged IndicNER entities: {merged}")
    return merged




def normalize_label(label: str) -> str:
    # Map IndicNER tags like "PER", "LOC", "ORG" to spaCy-style tags
    label_map = {
        "PER": "PERSON",
        "LOC": "GPE",
        "ORG": "ORG"
    }
    return label_map.get(label, label)

def mask_structured_pii(text: str) -> str:
    # Email masking
    email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
    text = email_pattern.sub("<EMAIL>", text)

    phone_spans = set()
    for region in ["IN", "US", "GB"]:
        for match in phonenumbers.PhoneNumberMatcher(text, region):
            phone_spans.add((match.start, match.end))

    for start, end in sorted(phone_spans, reverse=True):
        logger.info(f"Replacing phone span [{start}:{end}] with <PHONE>")
        text = text[:start] + "<PHONE>" + text[end:]
        print(f"Masked text so far (phone): {text}")

    logger.info(f"Masked text: {text}")

    #SSN_masking
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    text = re.sub(ssn_pattern, '[SSN_MASKED]', text)

    return text

def mask_pii(text: str) -> str:
    logger.info(f"Input text for masking: {text}")

    spacy_entities = get_spacy_entities(text)
    indic_entities = get_indic_entities(text)

    logger.info(f"spaCy entities: {spacy_entities}")
    logger.info(f"IndicNER entities: {indic_entities}")

    combined_entities = spacy_entities + indic_entities
    combined_entities.sort(key=lambda x: x.get("start", 0), reverse=True)
    

    for entity in combined_entities:
        start = entity.get("start")
        end = entity.get("end")
        label = entity["entity_group"]

        if start is not None and end is not None:
            span = text[start:end]
            pattern = re.compile(rf'\b\w*{re.escape(span)}\w*\b', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                logger.info(f"Replacing word span [{match.start()}:{match.end()}] with <{label}>")
                text = text[:match.start()] + f"<{label}>" + text[match.end():]
                print(f"Masked text so far: {text}")

    text = mask_structured_pii(text)
    logger.info(f"Masked text: {text}")
    return text
