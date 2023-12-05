# Inference commands
```
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir ./llama-2-7b --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
```

```
python -m torch.distributed.run --nproc_per_node 1 example_text_completion.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```

```
python example_text_completion.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```

# Inference prompts & results
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

===================================



# Fine-tuning commands
```
python train.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```

```
python alpaca_test.py --ckpt_dir /project/saifhash_1190/llama2-7b --tokenizer_path /project/saifhash_1190/llama2-7b/tokenizer.model
```


# Fine-tuning analysis
Total parameters: 6747639808
Trainable parameters: 8388608
Percentage of trainable parameters: 0.12%

GPU memory usage:
40402MiB / 40960MiB

Loss for every epoch:
Epoch 0: Average Loss = 1.325083038618483
Epoch 1: Average Loss = 1.0072221773462813
Epoch 2: Average Loss = 0.7484890972584303
Epoch 3: Average Loss = 0.5021964082850953
Epoch 4: Average Loss = 0.3443939249542737


# Fine-tuning prompts & results
Explain why a given topic is important. Environmental protection
> Environmental protection is essential for maintaining the health of our planet for future generations. Human activities such as emissions, farming, and deforestation are contributing to climate change, air and water pollution, reduction in biodiversity, and other environmental threats. By placing a priority on environmental protection, we can reduce the speed of environmental damage and conserve natural resources for our own and future generations' use.

==================================

Rewrite the following sentence without changing the meaning. The sales report was not accurate.
> The sales report was inaccurate.

==================================

Explain why the given definition is wrong. A mole is an animal that lives underground.
> This definition is wrong because moles are small, furry mammals that live above ground and are known for their distinctive long snouts and wide, black eyes. They are most commonly found in grassy areas and wooded areas.

==================================

Generate a password of 8 characters. 
> 1q2w3e4r s5t6u7v8

==================================


# Result from original model
Explain why a given topic is important. Environmental protection
> is a major issue that is faced by the world today. Many people are not aware of the negative impacts that they have on the environment. Environmental protection is important because it will help to conserve natural resources, improve the quality of the environment and also improve human health. Environmental protection is the process of protecting the environment from pollution and degradation. It involves the use of technology to reduce pollution and improve the quality of the environment.
The importance of environmental protection is that it will help to conserve natural resources. Natural resources are the resources that are found in nature. These resources are used by humans for their survival. Natural resources include water, air, land, forests, minerals and animals. Natural resources are important because they are used by humans for their survival. They are also important because they are used by animals for their survival.
The importance of environmental protection is that it will help to improve the quality of the environment. The quality of the environment is the condition of the environment. The quality of the environment is important because it will help to improve the quality of life. The quality of the environment is important because it will help to improve the

==================================

Rewrite the following sentence without changing the meaning. The sales report was not accurate.
> 
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate.
The sales report was not accurate

==================================

Explain why the given definition is wrong. A mole is an animal that lives underground.
> 
An animal that lives underground is an animal that lives in a hole in the ground.
Explain why the given definition is wrong. A mole is an animal that lives in a hole in the ground.
An animal that lives in a hole in the ground is an animal that lives underground.
Q: Explain why the given definition is wrong. A mole is an animal that lives underground.
A mole is an animal that lives underground.
Q: Explain why the given definition is wrong. A mole is an animal that lives in a hole in the ground.
An animal that lives in a hole in the ground is an animal that lives underground.
Q: Explain why the given definition is wrong. A mole is an animal that lives in a hole in the ground.
Q: Explain why the given definition is wrong. A mole is an animal that lives underground.
Q: Explain why the given definition is wrong. A mole is an animal that lives in a hole in the ground.
Q: Explain why the given definition is wrong. A mole is an animal that lives in a hole

==================================

Generate a password of 8 characters.
> 

I have a password field in a form, and I want to generate a password of 8 characters.

I'm trying this code, but it is not working:

\begin{code}
function generatePassword() {
  var length = 8;
  var chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  var result = "";
  for (var i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
document.getElementById("password").value = generatePassword();
\end{code}

I'm trying to generate a password of 8 characters.
But I'm getting this error:
Uncaught TypeError: Cannot read property 'length' of undefined

Comment: Your code is missing the closing `)` on the `return` statement
Comment: `var length = 8;
 

==================================