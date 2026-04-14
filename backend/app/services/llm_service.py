import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.core.config import ADAPTER_PATH, BASE_MODEL_PATH, MAX_NEW_TOKENS


STOP_PATTERNS = [
    r"\nHuman:",
    r"\nUser:",
    r"\nContext:",
    r"\nTitle:",
    r"\nTopic:",
    r"\nQuestion:",
    r"\nAnswer:",
    r"\n###",
    r"\[Document\]",
    r"Human:",
    r"User:",
    r"Context:",
    r"Title:",
    r"Topic:",
    r"Question:",
    r"Answer:",
    r"###",
]


class LLMService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            is_trainable=False,
        )
        self.model.eval()

    def clean_model_output(self, decoded_text: str) -> str:
        text = decoded_text.strip()

        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()

        cut_positions = []
        for pattern in STOP_PATTERNS:
            match = re.search(pattern, text)
            if match:
                cut_positions.append(match.start())

        if cut_positions:
            text = text[:min(cut_positions)].strip()

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_model_output(decoded)