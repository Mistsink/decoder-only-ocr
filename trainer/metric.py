import evaluate

from config.config import Config


def normalized_edit_distance(wer_metric, preds, labels):
    total_distance = 0
    total_length = 0
    for pred, label in zip(preds, labels):
        distance = wer_metric.compute(predictions=[pred], references=[label])
        length = len(label)
        total_distance += distance * length
        total_length += length
    return total_distance / total_length if total_length > 0 else 0


class OCRMetric:
    def __init__(self, cfg: Config, tokenizer):
        self.cer_metric = evaluate.load("cer", cache_dir=cfg.cache_dir)
        self.wer_metric = evaluate.load("wer", cache_dir=cfg.cache_dir)
        self.tokenizer = tokenizer

    def __call__(self, pred):
        preds, labels = pred.predictions[0], pred.label_ids
        labels[labels==-100] = self.tokenizer.eos_token_id
        # Convert predicted token IDs to strings
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # CER
        cer = self.cer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        # NED (Normalized Edit Distance)
        ned = normalized_edit_distance(self.wer_metric, decoded_preds, decoded_labels)

        # Accuracy
        correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
        acc = correct / len(decoded_labels) if len(decoded_labels) > 0 else 0

        return {
            "cer": cer,
            "ned": ned,
            "acc": acc,
        }
