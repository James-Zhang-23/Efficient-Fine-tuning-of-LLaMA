### Inference example
```
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir ./llama-2-7b --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
```

```
python -m torch.distributed.run --nproc_per_node 1 example_text_completion.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```