import numpy as np
from datasets import load_metric

# Load the metrics
cer_metric = load_metric("cer")
wer_metric = load_metric("wer")


def normalized_edit_distance(preds, labels):
    total_distance = 0
    total_length = 0
    for pred, label in zip(preds, labels):
        distance = wer_metric.compute(predictions=[pred], references=[label])[
            "wer"
        ]  # Using WER metric to calculate edit distance
        length = len(label)
        total_distance += distance * length
        total_length += length
    return total_distance / total_length if total_length > 0 else 0


class OCRMetric:
    def __init__(self, tokenizer):
        self.cer_metric = load_metric("cer")
        self.wer_metric = load_metric("wer")
        self.tokenizer = tokenizer

    def __call__(self, pred):
        preds, labels = pred.predictions, pred.label_ids
        # Convert predicted token IDs to strings
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # CER
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # NED (Normalized Edit Distance)
        ned = normalized_edit_distance(decoded_preds, decoded_labels)

        # Accuracy
        correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
        acc = correct / len(decoded_labels) if len(decoded_labels) > 0 else 0

        return {
            "cer": cer,
            "ned": ned,
            "acc": acc,
        }
