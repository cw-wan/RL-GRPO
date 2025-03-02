from trl import GRPOConfig, GRPOTrainer
import json
from utils import _load_dataset, _batch_parse_answers

MODELS = {
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
}

DATASETS = {
    "GSM8K": lambda ratio: _load_dataset("openai/gsm8k", "main", split="train", ratio=ratio),
}

CONVERTION = {
    "GSM8K": lambda ds: ds.rename_columns({"question": "prompt", "answer": "completion"})
}

with open("config.json", "r") as config_file:
    config = json.load(config_file)

model_name = config["model"]
dataset_name = config["dataset"]
train_ratio = config["train_ratio"]

dataset = DATASETS[dataset_name](train_ratio)
dataset = CONVERTION[dataset_name](dataset)
dataset = dataset.map(lambda example: {"ground_truth": _batch_parse_answers([example["completion"]])[0]})

print(dataset[0])

def reward_func(completions, ground_truth, **kwargs):
    pred, gt = _batch_parse_answers(completions), ground_truth
    return [1.0 if p == g else 0.0 for p, g in zip(pred, gt)]


training_args = GRPOConfig(output_dir=f"{config["model"]}-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model=MODELS[model_name],
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
