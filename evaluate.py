import argparse
import json

from tqdm import tqdm
from models_QWen import *

from QWen_utils import _load_dataset, _batch_parse_answers

MODELS = {
    "Qwen2.5-0.5B-Instruct": Qwen("Qwen/Qwen2.5-0.5B-Instruct"),
    "Qwen2.5-0.5B-Instruct-GRPO": lambda ckpt: Qwen("Qwen/Qwen2.5-0.5B-Instruct", load_checkpoint=True,
                                                    checkpoint_path=ckpt),
    "Qwen2.5-0.5B-Instruct-SFT": lambda ckpt: Qwen("Qwen/Qwen2.5-0.5B-Instruct", load_checkpoint=True,
                                                    checkpoint_path=ckpt),
}

DATASETS = {
    "GSM8K": lambda ratio: _load_dataset("openai/gsm8k", "main", split="test", ratio=ratio),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys())
    parser.add_argument("--model", choices=MODELS.keys())
    parser.add_argument("--batch_size", required=False, default=4)
    parser.add_argument("--k", type=int, required=False, default=4)
    parser.add_argument("--test_ratio", type=int, required=False, default=10)
    parser.add_argument("--ckpt", required=False, default="")

    args = parser.parse_args()

    if not args.ckpt:
        model = MODELS[args.model]
    else:
        model = MODELS[args.model](ckpt=args.ckpt)
    dataset = DATASETS[args.dataset](args.test_ratio)

    pass_at_one = []

    total_batches = len(dataset) // args.batch_size

    bar = tqdm(
        enumerate(dataset.iter(batch_size=args.batch_size)),
        total=total_batches
    )

    cases = []

    for _, data in bar:
        bar.set_description("Evaluating {} on {}".format(args.model, args.dataset))
        gt = _batch_parse_answers(data["answer"])
        prediction_samples = []
        model_answers = []
        for _ in range(args.k):
            predictions = model.batch_inference(data["question"])
            model_answers.append(predictions)
            parsed_pred = _batch_parse_answers(predictions)
            prediction_samples.append(parsed_pred)
        for q, a, mas in zip(data["question"], data["answer"], list(zip(*model_answers))):
            cases.append({
                "Question": q,
                "Ground Truth": a,
                "Model Predictions": mas,
            })
            with open(f"{args.model}-cases.json", "w") as f:
                json.dump(cases, f)
        prediction_samples = list(zip(*prediction_samples))
        for y, x in zip(gt, prediction_samples):
            print(y, x)
            cnt = 0
            for pred in x:
                if pred == y:
                    cnt += 1
            pass_at_one.append(cnt / len(x))

    print("Pass@1: {:.2f}%".format(100 * sum(pass_at_one) / len(pass_at_one)))
