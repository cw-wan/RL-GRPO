import argparse
from tqdm import tqdm
from models import *

from utils import _load_dataset, _parse_answer

MODELS = {
    "Qwen2.5-0.5B-Instruct": Qwen("Qwen/Qwen2.5-0.5B-Instruct"),
}

DATASETS = {
    "GSM8K": lambda ratio: _load_dataset("openai/gsm8k", "main", split="test", ratio=ratio),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys())
    parser.add_argument("--model", choices=MODELS.keys())
    parser.add_argument("--batch_size", required=False, default=4)
    parser.add_argument("--test_ratio", type=int, required=False, default=10)

    args = parser.parse_args()

    model = MODELS[args.model]
    dataset = DATASETS[args.dataset](args.test_ratio)

    correct_cnt = 0
    total_questions = len(dataset)

    total_batches = total_questions // args.batch_size

    bar = tqdm(
        enumerate(dataset.iter(batch_size=args.batch_size)),
        total=total_batches
    )

    for _, data in bar:
        bar.set_description("Evaluating {} on {}".format(args.model, args.dataset))
        gt = _parse_answer(data["answer"])
        predictions = model.batch_inference(data["question"])
        parsed_pred = _parse_answer(predictions)
        for x, y in zip(gt, parsed_pred):
            correct_cnt += 1 if x == y else 0

    print("Solve rate: {:.2f}%".format(100 * correct_cnt / total_questions))
