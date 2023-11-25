import json
import fire
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from train import LlamaAlpaca


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = LlamaAlpaca.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    generator.model.load_state_dict(torch.load('model_state_dict.pth'))
    generator.model.eval()
    prompts: List[str] = [
        # Zero shot prompts
        "What are the three primary colors?",
        "How can we reduce air pollution?",
        "What is the capital of France?",

        # Few shot prompts (providing a few examples before asking model to complete more);
        # tbd
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
