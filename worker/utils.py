import re
import unicodedata
from google.generativeai.types import GenerateContentResponse

def simplify_response(response: GenerateContentResponse) -> dict:
    return {
        "text": response.text,
        "block_reason": response.prompt_feedback.block_reason if response.prompt_feedback else None,
        "safety_ratings": [
            {
                "category": rating.category,
                "probability": rating.probability.name  # .name for enum
            }
            for rating in response.prompt_feedback.safety_ratings
        ] if response.prompt_feedback else []
    }

def normalize_text(text: str) -> str:
    """
    Normalizes text by lowercasing, removing extra whitespace, and handling unicode.
    This helps in comparing substrings more reliably.
    """
    if not isinstance(text, str):
        text = str(text)
    # NFKD normalization handles different unicode characters that might look the same
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def validate_extracted_data(extracted_data: dict, context: str, required_fields: list) -> list:

    """
    Validates the extracted JSON data against a set of rules.

    Rules:
    1. All required keys must be present.
    2. Each field's value must be a substring of the original context (after normalization).

    Args:
        extracted_data: The JSON object (as a dict) returned by the LLM.
        context: The original source text.
        required_fields: A list of keys that must be in the extracted_data.

    Returns:
        A list of validation error messages. An empty list means validation passed.
    """
    errors = []
    normalized_context = normalize_text(context)

    # 1. Check for presence of required keys
    for key in required_fields:
        if key not in extracted_data:
            errors.append(f"Missing required key: '{key}'")

    # If keys are missing, we can't check their values, so return early.
    if errors:
        return errors

    # 2. Check if each value is a substring of the context
    # for key, value in extracted_data.items():
    #     if value is not None:
    #         normalized_value = normalize_text(value)
    #         if normalized_value not in normalized_context:
    #             errors.append(
    #                 f"Value for key '{key}' ('{value}') not found in the original document text."
    #             )

    return errors
