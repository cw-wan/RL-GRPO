import re
from datasets import load_dataset


def _load_dataset(name, subset, split, ratio):
    return load_dataset(name, subset, split=f"{split}[:{ratio}%]")


def _batch_parse_answers(ans):
    pattern = r"\d+(?:/\d+|\.\d+|(?:,\d+)+)?"  # Matches integers, decimals, and fractions

    results = []
    for text in ans:
        numbers = re.findall(pattern, text)  # Find all numbers in the text
        results.append(numbers[-1].replace(",", "") if numbers else None)  # Take the last one as the final result

    return results
