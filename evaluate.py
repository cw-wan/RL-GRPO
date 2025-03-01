from datasets import load_dataset

MODELS = {}

DATASETS = {
    "GSM8K": load_dataset("openai/gsm8k", "main", split="test"),
}
