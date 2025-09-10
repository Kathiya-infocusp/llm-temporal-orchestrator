import os 
import re
import json
import unicodedata

from datetime import datetime
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import  f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from google.generativeai.types import GenerateContentResponse

REQUIRED_FIELDS = [
    "INVOICE_NUMBER", "DATE_OF_ISSUE", "BILLED_TO", "ADDRESS",
    "ITEM_DESCRIPTION", "QTY", "UNIT_PRICE", "AMOUNT",
    "TOTAL_AMOUNT", "BANK_NAME", "ACCOUNT_NAME", "ACCOUNT_NUMBER"
]

def save_json_artifact(data: dict, output_dir: str, filename: str):
    """
    Saves a dictionary as a JSON file in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return file_path

def match(gt: str, pred: str | list[str]) -> bool:
    if gt is None and pred is None:
        return True
    if gt is None: #or pred is None:
        return True
    if pred is None: 
        return False
    if isinstance(pred, list):
        return gt.strip().lower() in [_.strip().lower() for _ in pred]
    
    return gt.strip().lower() == pred.strip().lower()

def normalize_fields(entry: Dict) -> Dict:
    return {field: entry.get(field, None) for field in REQUIRED_FIELDS}

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

def validate_extracted_data(extracted_data: dict, context: str, required_fields: List[str] = REQUIRED_FIELDS) -> list:
    """
    Validates the extracted JSON data against a set of rules.

    Rules:
    1. All required keys must be present.
    2. Each field's value must be a substring of the original context (after normalization).

    Args:
        extracted_data: The JSON object (as a dict) returned by the LLM.
        context: The original source text.

    Returns:
        A list of validation error messages. An empty list means validation passed.
        
    """
    errors = []
    normalized_context = normalize_text(context)
    normalized_input = {key.upper(): key for key in extracted_data} 

    expected_keys = {
        "INVOICE_NUMBER": ["INVOICE_NUMBER"],
        "DATE_OF_ISSUE": ["DATE_OF_ISSUE", "DATE"],
        "DATE": ["DATE_OF_ISSUE", "DATE"],
        "SERVICE_DATE": ["SERVICE_DATE"],
        "BILLED_TO": ["BILLED_TO", "ISSUED_TO"],
        "ISSUED_TO": ["BILLED_TO", "ISSUED_TO"],
        "ADDRESS": ["ADDRESS"],
        "PHONE": ["PHONE"],
        "EMAIL": ["EMAIL"],
        "ITEM_DESCRIPTION": ["ITEM_DESCRIPTION"],
        "QTY": ["QTY", "QUANTITY"],
        "QUANTITY": ["QTY", "QUANTITY"],
        "UNIT_PRICE": ["UNIT_PRICE", "PRICE"],
        "PRICE": ["UNIT_PRICE", "PRICE"],
        "AMOUNT": ["AMOUNT", "TOTAL"],
        "TOTAL": ["AMOUNT", "TOTAL"],
        "SUBTOTAL": ["SUBTOTAL"],
        "TAX": ["TAX"],
        "TOTAL_AMOUNT": ["TOTAL_AMOUNT", "GRAND_TOTAL"],
        "GRAND_TOTAL": ["TOTAL_AMOUNT", "GRAND_TOTAL"],
        "BANK_NAME": ["BANK_NAME"],
        "ACCOUNT_NAME": ["ACCOUNT_NAME"],
        "ACCOUNT_NUMBER": ["ACCOUNT_NUMBER"],
        "PAYMENT_TERMS": ["PAYMENT_TERMS"],
        "PAYMENT_DATE": ["PAYMENT_DATE"]
    }

    # 1. Check for presence of required keys
    for key in required_fields:
        if key in expected_keys.keys():
            synonyms = expected_keys[key]
        else:
            synonyms=[key]
        match_found = False
        for synonym in synonyms:
            synonym_upper = synonym.upper()
            if synonym_upper in normalized_input:
                match_found = True
                break


        if not match_found:
            errors.append(f"Missing required key: '{key}'")

    # If keys are missing, we can't check their values, so return early.
    if errors:
        return errors

    # TODO update this section for better metrics
    # 2. Check if each value is a substring of the context
    for key, values in extracted_data.items():
        if values :
            if isinstance(values, str):
                values = [values]
            for value in values:
                if value == 'None':
                    continue
                normalized_value = normalize_text(value)
                if normalized_value not in normalized_context:
                    errors.append(
                        f"Value for key '{key}' ('{value}') not found in the original document text."
                    )

    return errors

def evaluate(gt_data: List[Dict], pred_data: List[Dict]) -> Dict:
    assert len(gt_data) == len(pred_data), "Mismatch in number of documents."

    field_metrics = {field: {'tp': 0, 'fp': 0, 'fn': 0} for field in REQUIRED_FIELDS}
    total_documents = len(gt_data)
    exact_match_count = 0

    for gt_raw, pred_raw in zip(gt_data, pred_data):
        gt = normalize_fields(gt_raw)
        pred = normalize_fields(pred_raw)

        all_fields_match = True

        for field in REQUIRED_FIELDS:
            gt_val = gt[field]
            pred_val = pred[field]

            if match(gt_val, pred_val):
                if gt_val is not None:
                    field_metrics[field]['tp'] += 1
            else:
                all_fields_match = False
                if pred_val is not None and gt_val is None:
                    field_metrics[field]['fp'] += 1
                elif pred_val is None and gt_val is not None:
                    field_metrics[field]['fn'] += 1
                else:
                    field_metrics[field]['fp'] += 1
                    field_metrics[field]['fn'] += 1

        if all_fields_match:
            exact_match_count += 1

    # Compute metrics
    metrics_output = {}
    macro_f1_total = 0.0

    for field, counts in field_metrics.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        macro_f1_total += f1

        metrics_output[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    macro_f1 = macro_f1_total / len(REQUIRED_FIELDS)
    doc_accuracy = exact_match_count / total_documents

    final_result = {
        "macro_f1": round(macro_f1, 4),
        "document_accuracy": round(doc_accuracy, 4),
        "exact_matches": exact_match_count,
        "documents_evaluated": total_documents,
        "field_metrics": metrics_output,
    }

    return final_result

def log_structured(workflow_id: str, activity: str, **kwargs):
    import sys
    try:
        log_dir = Path(f"./runs/{workflow_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "workflowId": workflow_id,
            "activity": activity,
            **kwargs
        }
        log_file = log_dir / "workflow.log"
        # Ensure the file exists first
        log_file.touch(exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()  # Ensure data is written immediately
        
    except Exception as e:
        # Fallback logging to stderr if file logging fails
        print(f"ERROR: Log directory path: {log_dir if 'log_dir' in locals() else 'undefined'}", file=sys.stderr)