import json
import fire
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from train import LlamaAlpaca
from llama import Llama


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
    # Load the pretrained state dictionary (without applying it to the model)
    # pretrained_dict = torch.load('/project/saifhash_1190/llama2-7b/consolidated.00.pth', map_location="cuda")
    # Load the model state as usual
    # generator.model.load_state_dict(pretrained_dict, strict=False)
    # load finetuned LoRA weight
    generator.model.load_state_dict(torch.load('lora-finetuned.pth',map_location="cuda"), strict=False)
    # Check for parameters that were not correctly loaded
    # for name, param in generator.model.named_parameters():
    #     if name in pretrained_dict and not torch.equal(pretrained_dict[name], param):
    #         print(f"Parameter '{name}' may not have been correctly loaded.")

    generator.model.eval()
    prompts: List[str] = [
        # Zero shot prompts
        "Give three tips for staying healthy.",

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
