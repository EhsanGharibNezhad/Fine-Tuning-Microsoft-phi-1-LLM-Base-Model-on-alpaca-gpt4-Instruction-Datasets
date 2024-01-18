# Project: Fine-Tuning Microsoft/phi-1 LLM Base Model on alpaca-gpt4 Instruction Datasets 

Key steps:
- Utilized PEFT/LoRA techniques for fine-tuning the Microsoft/phi-1_5 pre-trained model on alpaca-gpt4 datasets.

- Leveraged advanced training approaches to refine the model's capabilities, aiming for improved precision, recall,
and other relevant metrics.

For the fine-tuned model and parameters, please check: [HuggingFace](https://huggingface.co/ehsangharibnezhad/phi-1_5-finetuned-vicgalle-alpaca-gpt4)

![image](LoRA_overview.jpg "Figure 1: LoRa overview -- [Ref]([https://huggingface.co/ehsangharibnezhad/phi-1_5-finetuned-vicgalle-alpaca-gpt4](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms))")


### Figure Legend: LoRA Overview

    1. LoRA can be implemented as an adapter designed to enhance and expand the existing neural network layers. 
    It introduces an additional layer of trainable parameters (weights) while maintaining the original parameters in a frozen state. These trainable parameters possess a substantially reduced rank (dimension) compared to the dimensions of the original network. This is the mechanism through which LoRa simplifies and expedites the process of adapting the original models for domain-specific tasks. Now, let’s take a closer look at the components within the LORA adapter network.

    2. The pre-trained parameters of the original model (W) are frozen. During training, these weights will not be modified.

    3. A new set of parameters is concurrently added to the networks WA and WB. These networks utilize low-rank weight vectors, where the dimensions of these vectors are represented as dxr and rxd. Here, ‘d’ stands for the dimension of the original frozen network parameters vector, while ‘r’ signifies the chosen low-rank or lower dimension. The value of ‘r’ is always smaller, and the smaller the ‘r’, the more expedited and simplified the model training process becomes. Determining the appropriate value for ‘r’ is a pivotal decision in LoRA. Opting for a lower value results in faster and more cost-effective model training, though it may not yield optimal results. Conversely, selecting a higher value for ‘r’ extends the training time and cost, but enhances the model’s capability to handle more complex tasks.
    
    4. The results of the original network and the low-rank network are computed with a dot product, which results in a weight matrix of n dimension, which is used to generate the result.
    
    5. This result is then compared with the expected results (during training) to calculate the loss function and WA and WB weights are adjusted based on the loss function as part of backpropagation like standard neural networks.

Ref: 
- https://abvijaykumar.medium.com/fine-tuning-llm-parameter-efficient-fine-tuning-peft-lora-qlora-part-1-571a472612c4
- https://arxiv.org/pdf/2106.09685.pdf

### What Libraries? 
   - `!pip install -q loralib`
   - `!pip install -q git+https://github.com/huggingface/peft.git`
   
### More about LoRA parameters?
   - __LoRA Dimension / Rank of Decomposition r__: For each layer to be trained, the d × k weight update matrix ∆W is represented by a low-rank decomposition BA, where B is a d × r matrix and A is a r × k matrix. The rank of decomposition r is << min(d,k). The default of r is 8. A is initialized by random Gaussian numbers so the initial weight updates have some variation to start with. B is initialized by by zero so ∆W is zero at the beginning of training. 
   - __Alpha Parameter for LoRA Scaling lora_alpha__:  ∆W is scaled by α / r where α is a constant. When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if the initialization was scaled appropriately.
   
   - __Dropout Probability for LoRA Layers lora_dropout__: Dropout is a technique to reduce overfitting by randomly selecting neurons to ignore with a dropout probability during training. The contribution of those selected neurons to the activation of downstream neurons is temporally removed on the forward pass, and any weight updates are not applied to the neuron on the backward pass. The default of lora_dropout is 0.
   - __Bias Type for Lora bias__: Bias can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation. The default is None.
   
![image](LoRA_parameters2.jpg)

Ref: 
   - https://medium.com/@manyi.yim/more-about-loraconfig-from-peft-581cf54643db
   - https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
