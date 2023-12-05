# Final Report

## Section 5: LLaMA Training

### 5.8 Analysis

#### LoRA settings
lora_rank = 16
lora_alpha = 32, 
lora_dropout = 0.05

#### Trainable parameter count 
Total parameters: 6747639808
Trainable parameters: 8388608
Percentage of trainable parameters: 0.12%

#### GPU memory usage
40402MiB / 40960MiB

#### Gradient checkpoints

#### Training loss for every epoch:
Epoch 0: Average Loss = 1.325083038618483
Epoch 1: Average Loss = 1.0072221773462813
Epoch 2: Average Loss = 0.7484890972584303
Epoch 3: Average Loss = 0.5021964082850953
Epoch 4: Average Loss = 0.3443939249542737

#### Comprehensive analysis
For each of the techniques applied (low precision, gradient accumulation, gradient checkpointing, LoRA), analyze whether or not it can contribute to memory reduction (in terms of model weight, model activation, model gradient, and optimizer state), computational cost, and potential impact on model accuracy.

| Technique | Memory Reduction | Computational Cost Reduction | Impact on Accuracy |
| --- | --- | --- | --- |
| Low Precision | Yes | Yes | Yes |
| Gradient Accumulation | No | Yes | No |
| Gradient Checkpointing | No | Yes | No |
| LoRA | Reduce model gradient and optimizer state | Yes | Decrease accuracy compare to training on all parameters |


### Example prompts and results (fine-tuned model)

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


### Example prompts and results (original model)
Explain why a given topic is important. Environmental protection
> is a major issue that is faced by the world today. Many people are not aware of the negative impacts that they have on the environment. Environmental protection is important because it will help to conserve natural resources, improve the quality of the environment and also improve human health. Environmental protection is the process of protecting the environment from pollution and degradation. It involves the use of technology to reduce pollution and improve the quality of the environment.
The importance of environmental protection is that it will help to conserve natural resources. Natural resources are the resources that are found in nature. These resources are used by humans for their survival. Natural resources include water, air, land, forests, minerals and animals. Natural resources are important because they are used by humans for their survival. They are also important because they are used by animals for their survival.
The importance of environmental protection is that it will help to improve the quality of the environment. The quality of the environment is the condition of the environment. The quality of the environment is important because it will help to improve the quality of life. The quality of the environment is important because it will help to improve the

==================================

Rewrite the following sentence without changing the meaning. The sales report was not accurate.
> The sales report was not accurate.
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
> An animal that lives underground is an animal that lives in a hole in the ground.
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
> I have a password field in a form, and I want to generate a password of 8 characters.
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




