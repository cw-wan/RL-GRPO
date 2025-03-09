from transformers import AutoModelForCausalLM, AutoTokenizer
from QWen_utils import _load_dataset
from trl import SFTConfig, SFTTrainer

dataset = _load_dataset("openai/gsm8k", "main", split="train", ratio=30)
dataset = dataset.rename_columns({"question": "prompt", "answer": "completion"})

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="Qwen2.5-0.5B-Instruct-SFT", logging_steps=10,
                   logging_dir="Qwen2.5-0.5B-Instruct-SFT-log", max_seq_length=512),
)

trainer.train()
