### Inference example
```
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir ./llama-2-7b --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
```

```
python -m torch.distributed.run --nproc_per_node 1 example_text_completion.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```

# Prompts & Results
## Zero shot prompts
"What is machine learning?",
"To maintain a healthy work-life balance, it's essential to",

## Few shot prompts
"""Find synonyms for the following words:
quick => fast
happy => joyful
difficult => challenging
intelligent =>""",

"""Match the event to the year:

Moon Landing => 1969
Fall of the Berlin Wall => 1989
Invention of the Internet => 1983
Start of World War I =>""",

## Results
What is machine learning?
> How does it work?
A lot of people have heard about machine learning, but few know what it actually is. Machine learning is a type of artificial intelligence (AI) that allows computers to learn from data without being explicitly programmed.
Machine learning algorithms are used to make predictions based on patterns in data. For

==================================

To maintain a healthy work-life balance, it's essential to
> have a healthy lifestyle. It is not always easy to maintain a healthy lifestyle, especially when you have a busy schedule. However, it is important to make time for yourself and your health. This article will discuss how to maintain a healthy work-life balance with a healthy lifest

==================================

Find synonyms for the following words:
        quick => fast
        happy => joyful
        difficult => challenging
        intelligent =>
> smart
        silly => dumb
        simple => easy
        fast => quick
        loud => noisy
        dumb => silly
        scary => spooky
        noisy => loud
        quiet => silent
        smart => intelligent
        challenging => difficult
        easy =>

==================================

Match the event to the year:

        Moon Landing => 1969
        Fall of the Berlin Wall => 1989
        Invention of the Internet => 1983
        Start of World War I =>
> 1914
        Start of World War II => 1939
        Start of the Vietnam War => 1964
        Start of the Korean War => 1950
        Start of the Cold War => 1947
        Start of the Cuban

==================================