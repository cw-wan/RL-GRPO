from datasets import load_dataset
from models import *

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen("Qwen/Qwen2.5-0.5B-Instruct"),
}

DATASETS = {
    "GSM8K": load_dataset("openai/gsm8k", "main", split="test"),
}
