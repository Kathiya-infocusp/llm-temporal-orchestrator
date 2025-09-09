import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class FieldMetrics:
    exact_matches: int = 0
    normalized_matches: int = 0
    total_fields: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total_fields if self.total_fields > 0 else 0.0
    
    @property
    def normalized_match_rate(self) -> float:
        return self.normalized_matches / self.total_fields if self.total_fields > 0 else 0.0
    
    @property
    def precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def normalize_value(value: Any) -> str:
    """Normalize a value for comparison"""
    if value is None:
        return ""
    
    if isinstance(value, list):
        return " ".join(str(v).lower().strip() for v in value)
    
    text = str(value).lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation variations
    text = re.sub(r'[.,;:!?]', '', text)
    return text


def evaluate_field_extraction(predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate field extraction performance with multiple metrics
    
    Args:
        predictions: List of predicted field dictionaries
        ground_truths: List of ground truth field dictionaries
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    # Collect all unique field names
    all_fields = set()
    for gt in ground_truths:
        all_fields.update(gt.keys())
    
    # Initialize metrics for each field
    field_metrics = {field: FieldMetrics() for field in all_fields}
    overall_metrics = FieldMetrics()
    
    sample_results = []
    document_exact_matches = 0  # Count of documents with 100% exact match
    document_normalized_matches = 0  # Count of documents with 100% normalized match
    
    for pred, gt in zip(predictions, ground_truths):
        sample_exact = 0
        sample_normalized = 0
        sample_total = 0
        sample_details = {}
        
        for field in all_fields:
            gt_value = gt.get(field)
            pred_value = pred.get(field)
            
            # Skip if ground truth doesn't have this field
            if gt_value is None:
                continue
                
            sample_total += 1
            field_metrics[field].total_fields += 1
            overall_metrics.total_fields += 1
            
            # Exact match check (GT is never a list)
            exact_match = False
            if isinstance(pred_value, list):
                # Pred is list, GT is single - check if GT matches any element in pred list
                exact_match = str(gt_value) in [str(v) for v in pred_value]
            else:
                # Both are single values
                exact_match = pred_value == gt_value
            
            # Normalized match check (GT is never a list)
            normalized_match = False
            norm_gt = normalize_value(gt_value)
            if isinstance(pred_value, list):
                # Pred is list, GT is single - check if normalized GT matches any normalized pred element
                norm_pred_list = [normalize_value(v) for v in pred_value]
                normalized_match = norm_gt in norm_pred_list if norm_gt else False
            else:
                # Both are single values
                norm_pred = normalize_value(pred_value)
                normalized_match = norm_gt == norm_pred if norm_gt and norm_pred else False
            
            # Update metrics
            if exact_match:
                sample_exact += 1
                field_metrics[field].exact_matches += 1
                overall_metrics.exact_matches += 1
            
            if normalized_match:
                sample_normalized += 1
                field_metrics[field].normalized_matches += 1
                overall_metrics.normalized_matches += 1
            
            # F1 score components
            if pred_value is not None and gt_value is not None:
                if normalized_match:
                    field_metrics[field].true_positives += 1
                    overall_metrics.true_positives += 1
                else:
                    field_metrics[field].false_positives += 1
                    overall_metrics.false_positives += 1
            elif pred_value is not None and gt_value is None:
                field_metrics[field].false_positives += 1
                overall_metrics.false_positives += 1
            elif pred_value is None and gt_value is not None:
                field_metrics[field].false_negatives += 1
                overall_metrics.false_negatives += 1
            
            # Store details for incorrect predictions
            if not normalized_match:
                sample_details[field] = {
                    'ground_truth': gt_value,
                    'prediction': pred_value,
                    'exact_match': exact_match,
                    'normalized_match': normalized_match
                }
        
        # Calculate document-level match rates
        exact_match_rate = sample_exact / sample_total if sample_total > 0 else 0.0
        normalized_match_rate = sample_normalized / sample_total if sample_total > 0 else 0.0
        
        # Check if document is perfectly matched
        if exact_match_rate == 1.0:
            document_exact_matches += 1
        if normalized_match_rate == 1.0:
            document_normalized_matches += 1
        
        sample_results.append({
            'exact_matches': sample_exact,
            'normalized_matches': sample_normalized,
            'total_fields': sample_total,
            'exact_match_rate': exact_match_rate,
            'normalized_match_rate': normalized_match_rate,
            'incorrect_fields': sample_details
        })
    
    # Calculate document-level statistics
    total_documents = len(predictions)
    avg_exact_match_rate = sum(sample['exact_match_rate'] for sample in sample_results) / total_documents if total_documents > 0 else 0.0
    avg_normalized_match_rate = sum(sample['normalized_match_rate'] for sample in sample_results) / total_documents if total_documents > 0 else 0.0
    
    # Compile results
    results = {
        'overall_metrics': {
            'total_samples': len(predictions),
            'total_fields_evaluated': overall_metrics.total_fields,
            'exact_match_accuracy': overall_metrics.exact_match_rate,
            'normalized_match_accuracy': overall_metrics.normalized_match_rate,
            'precision': overall_metrics.precision,
            'recall': overall_metrics.recall,
            'f1_score': overall_metrics.f1_score
        },
        'document_level_metrics': {
            'total_documents': total_documents,
            'documents_exact_match': document_exact_matches,
            'documents_normalized_match': document_normalized_matches,
            'document_exact_match_rate': document_exact_matches / total_documents if total_documents > 0 else 0.0,
            'document_normalized_match_rate': document_normalized_matches / total_documents if total_documents > 0 else 0.0,
            'avg_exact_match_percentage_per_doc': avg_exact_match_rate,
            'avg_normalized_match_percentage_per_doc': avg_normalized_match_rate
        },
        'field_level_metrics': {
            field: {
                'exact_match_rate': metrics.exact_match_rate,
                'normalized_match_rate': metrics.normalized_match_rate,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'total_occurrences': metrics.total_fields
            }
            for field, metrics in field_metrics.items()
        },
        'sample_level_results': sample_results
    }
    
    return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print a formatted summary of evaluation results"""
    overall = results['overall_metrics']
    document = results['document_level_metrics']
    
    print("=== FIELD EXTRACTION EVALUATION SUMMARY ===")
    print(f"Total Documents: {document['total_documents']}")
    print(f"Total Fields Evaluated: {overall['total_fields_evaluated']}")
    print()
    print("DOCUMENT-LEVEL PERFORMANCE:")
    print(f"  Documents with 100% Exact Match: {document['documents_exact_match']}/{document['total_documents']} ({document['document_exact_match_rate']:.3f})")
    print(f"  Documents with 100% Normalized Match: {document['documents_normalized_match']}/{document['total_documents']} ({document['document_normalized_match_rate']:.3f})")
    print(f"  Average Exact Match % per Document: {document['avg_exact_match_percentage_per_doc']:.3f}")
    print(f"  Average Normalized Match % per Document: {document['avg_normalized_match_percentage_per_doc']:.3f}")
    print()
    print("FIELD-LEVEL PERFORMANCE:")
    print(f"  Overall Exact Match Accuracy: {overall['exact_match_accuracy']:.3f}")
    print(f"  Overall Normalized Match Accuracy: {overall['normalized_match_accuracy']:.3f}")
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall: {overall['recall']:.3f}")
    print(f"  F1 Score: {overall['f1_score']:.3f}")
    print()
    
    print("TOP PERFORMING FIELDS:")
    field_metrics = results['field_level_metrics']
    sorted_fields = sorted(
        [(field, metrics) for field, metrics in field_metrics.items() if metrics['total_occurrences'] > 0],
        key=lambda x: x[1]['f1_score'],
        reverse=True
    )
    
    for field, metrics in sorted_fields[:10]:  # Show top 10
        print(f"  {field}: F1={metrics['f1_score']:.3f}, Exact={metrics['exact_match_rate']:.3f}, Norm={metrics['normalized_match_rate']:.3f} ({metrics['total_occurrences']} occurrences)")