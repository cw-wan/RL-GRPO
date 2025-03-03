from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Qwen:
    def __init__(self,
                 model_name,
                 load_checkpoint=False,
                 checkpoint_path=None, ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not load_checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype="auto",
                device_map="auto"
            )
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def inference(self, p):
        messages = [
            {"role": "system",
             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant to solve math problems. Your final answer should be wrapped within \\boxed{}"},
            {"role": "user", "content": p},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def batch_inference(self, questions):
        message_batch = [[{"role": "user", "content": q}] for q in questions]
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)
        generated_ids_batch = self.model.generate(
            **model_inputs_batch,
            max_new_tokens=512,
        )
        generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
        response_batch = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
        return response_batch


if __name__ == "__main__":
    model = Qwen("Qwen2.5-0.5B-Instruct", load_checkpoint=True,
                 checkpoint_path="Qwen2.5-0.5B-Instruct-GRPO/checkpoint-6000")
    question = "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
    answer = model.inference(question)
    print(answer)
