import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import DEFAULT_MODEL_NAME
import numpy as np


class Conversation:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, block_size=128, num_seq=3, device="cuda"):
        self.num_seq = num_seq
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def prepare(self, prompt):
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def generate_text(self, encoded_prompt):
        with torch.no_grad():
            output = self.model.generate(encoded_prompt,
                                         max_length=self.block_size,
                                         do_sample=True,
                                         top_k=35,
                                         top_p=0.85,
                                         temperature=1.0,
                                         num_return_sequences=self.num_seq,
                                         eos_token_id=2,
                                         pad_token_id=0)
        return output

    def __call__(self, prompt, *args, **kwargs):
        encoded_prompt = self.prepare(prompt)
        output = self.generate_text(encoded_prompt)

        idx = np.random.randint(self.num_seq)
        output = output[idx]
        output = output[encoded_prompt.shape[1]:]
        decoded = self.tokenizer.decode(output, skip_special_tokens=True)

        return decoded.replace("\n", "").replace("-", "").strip()


if __name__ == "__main__":
    chat = Conversation("pankratozzi/rugpt3small_based_on_gpt2-finetuned-for-chat")
    chat("Почему ты не входишь в открытую дверь?")
