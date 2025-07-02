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

    # Aadhaar masking
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    text = re.sub(aadhaar_pattern, '[AADHAAR_MASKED]', text)

    # PAN masking
    pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
    text = re.sub(pan_pattern, '[PAN_MASKED]', text)

    # IFSC Code masking
    ifsc_pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
    text = re.sub(ifsc_pattern, '[IFSC_MASKED]', text)

    # Bank Account Number masking
    account_pattern = r'\b\d{9,18}\b'
    text = re.sub(account_pattern, '[BANK_ACCOUNT_MASKED]', text)

    # Credit/Debit Card Number masking
    card_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    text = re.sub(card_pattern, '[CARD_MASKED]', text)

    # Date of Birth masking (basic patterns)
    dob_patterns = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',        # 21/07/1995 or 21-07-1995
        r'\b\d{4}-\d{2}-\d{2}\b',              # 1995-07-21
        r'\b\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b',  # 21st July 1995
    ]
    for pattern in dob_patterns:
        text = re.sub(pattern, '[DOB_MASKED]', text)

    # UK NINO masking
    nino_pattern = r'\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b'
    text = re.sub(nino_pattern, '[NINO_MASKED]', text)

    # Address fallback (optional regex-based heuristic)
    address_patterns = [
        r'\b(?:Flat|House|Apt|Apartment|Villa|Building|Block|Sector|Street|Road|Colony|Lane|Nagar|Area|Vihar|Bagh|Enclave|Galli|Gali|Society|Floor|Tower)[\w\s,./-]{3,50}',
        r'\b(?:I\s+live\s+at|My\s+address\s+is)\s+[A-Za-z0-9\s,./-]{5,50}',  # context-based
        r'\b\d{1,4}[A-Za-z]?[/\-]?\d{1,4}[A-Za-z]?\s+[A-Z][a-z]{2,30}\b', 
    ]
    for pattern in address_patterns:
        text = re.sub(pattern, '[ADDRESS_MASKED]', text, flags=re.IGNORECASE)

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
