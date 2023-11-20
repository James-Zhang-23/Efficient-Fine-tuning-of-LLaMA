import json
import os
import sys
import time
import fire
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# Set environment variables for distributed training
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def train(
        self,
        prompt_tokens: List[List[int]],
        target_tokens: List[List[int]],
        epochs: int = 3,
        learning_rate: float = 5e-5
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()

        params = self.model.params
        pad_id = self.tokenizer.pad_id

        for epoch in range(epochs):
            for input, target in zip(prompt_tokens, target_tokens):
                total_len = 256
                optimizer.zero_grad()

                input_tensor = torch.full((1, total_len), pad_id, dtype=torch.long, device="cuda")
                input_tensor[0,: len(input)] = torch.tensor(input, dtype=torch.long, device="cuda")

                target_tensor = torch.tensor(target, dtype=torch.long, device="cuda").unsqueeze(0)

                prev_pos = 0
                eos_reached = torch.tensor([False] * 1, device="cuda")
                input_text_mask = input_tensor != pad_id

                target_pos = 0

                for cur_pos in range(len(input), total_len):
                    logits = self.model.forward(input_tensor[:, 0:cur_pos], prev_pos)

                    # Select the corresponding target token
                    current_target = target_tensor[:, target_pos]
                    
                    pred_token_logits = logits[:, -1, :]
                    loss = F.cross_entropy(pred_token_logits.view(-1, pred_token_logits.size(-1)), current_target.view(-1))
                    loss.backward()
                    optimizer.step()

# not finished yet
###############################################################################
                    next_token = torch.argmax(logits[:, -1], dim=-1)
                    next_token = next_token.reshape(-1)
                    # only replace token if prompt has already been generated
                    next_token = torch.where(
                        input_text_mask[:, cur_pos], input_tensor[:, cur_pos], next_token
                    )
                    input_tensor[:, cur_pos] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (
                        next_token == self.tokenizer.eos_id
                    )
                    prev_pos = cur_pos
                    target_pos += 1
                    if all(eos_reached):
                        break
###############################################################################


                
def load_and_process_dataset(json_path: str, tokenizer: Tokenizer) -> List[Tuple[List[int], List[int]]]:
    with open(json_path, 'r') as file:
        data = json.load(file)

    input_tokens = []
    target_tokens = []
    # read only first 200 examples
    data = data[:200]

    for item in data:
        input = tokenizer.encode(item['instruction'] + item['input'], bos=True, eos=False)
        target = tokenizer.encode(item['output'], bos=True, eos=True)
        input_tokens.append(input)
        target_tokens.append(target)

    return input_tokens, target_tokens


def main(
    ckpt_dir: str = './llama-2-7b',
    tokenizer_path: str = 'tokenizer.model',
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    json_dataset_path: str = 'alpaca_data.json',
):

    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompt_tokens, target_tokens = load_and_process_dataset(json_dataset_path, model.tokenizer)
    # model.train(prompt_tokens, target_tokens, epochs=epochs, learning_rate=learning_rate)


if __name__ == "__main__":
    fire.Fire(main)

