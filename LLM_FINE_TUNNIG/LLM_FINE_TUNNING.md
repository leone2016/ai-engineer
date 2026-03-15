# LLM Fine-Tuning Course Notes

## Course Topics

This course covers the following concepts and tools used in **Large Language Model (LLM) fine-tuning**:

- Supervised Fine-Tuning (SFT)
- Instruction Dataset Training
- Preference Alignment
- Hugging Face Ecosystem
- LLaMA Factory
- Unsloth
- Axolotl
- Small Language Model (SLM) Fine-Tuning
- Multimodal Fine-Tuning
- Embedding Fine-Tuning
- End-to-End Practical Implementation

Reference:
 https://sunnysavita10.github.io

------

# Fine-Tuning LLMs with the Hugging Face Ecosystem

Large Language Models (LLMs) are based on the **Transformer architecture**.

These models are typically trained through a **multi-stage training pipeline**.

------

# LLM Training Pipeline

The training process of an LLM generally consists of the following stages:

### 1. Unsupervised Pretraining

Also called **self-supervised learning**.

During this stage, the model learns general language patterns from large-scale text datasets.
 Typical objectives include:

- Next-token prediction
- Masked language modeling

This stage requires **massive datasets and large-scale GPU clusters**.

------

### 2. Supervised Fine-Tuning (SFT)

After pretraining, the model is fine-tuned using **labeled instruction datasets** so it can follow human instructions more effectively.

Fine-tuning can be performed at different levels depending on computational resources.

------

# Fine-Tuning Strategies

## 1. Full Fine-Tuning

In **Full Fine-Tuning**, all model parameters are updated:

- Weights
- Biases

### Characteristics

- Requires **large GPU clusters**
- High **VRAM consumption**
- High **training cost**
- Best performance for specialized tasks

------

## 2. Partial Fine-Tuning

Instead of training the entire model, only some layers are updated.

This approach was common in **early NLP models** such as:

- BART
- T5

### Traditional Approach

Two common strategies:

1. **Freeze most layers and retrain the final output layer**
2. **Freeze early layers and retrain later layers**

This reduces computational cost while still adapting the model to a new task.

------

# Parameter Efficient Fine-Tuning (PEFT)

PEFT methods allow training large models with **very limited computational resources**.

Advantages:

- Can run on **single GPU setups**
- Requires **less VRAM**
- Faster training
- Lower cost

PEFT methods modify only a **small subset of parameters** while keeping the original model mostly frozen.

------

## Popular PEFT Techniques

### LoRA (Low-Rank Adaptation)

LoRA adds small trainable matrices to attention layers.

Benefits:

- Very memory efficient
- Widely used in LLM fine-tuning

#### QLoRA

QLoRA combines:

- **Quantization**
- **LoRA**

Benefits:

- Lower precision models
- Extremely memory efficient
- Enables fine-tuning large models on consumer GPUs

------

### DoRA (Weight-Decomposed Low-Rank Adaptation)

An improved version of LoRA that separates magnitude and direction of weights.

Provides **better performance stability** in some scenarios.

------

### Adapter Layers

Small neural modules inserted between layers of a pretrained model.

Only these adapters are trained during fine-tuning.

------

### BitFit

A minimal fine-tuning approach where **only bias parameters** are updated.

Very lightweight but sometimes less expressive.

------

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

A PEFT technique that scales activations inside the network to adapt the model to new tasks.

------

### Prefix Tuning

Instead of modifying model weights, **trainable prefix vectors** are added to the attention mechanism.

------

### Prompt Tuning

The model is conditioned using **learnable prompts** added to the input embeddings.

```shell
                        ┌────────────────────────────┐
                        │        Raw Text Data       │
                        │  (Books, Web, Articles)   │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                     ┌────────────────────────────────┐
                     │   1. Unsupervised Pretraining  │
                     │   (Self-Supervised Learning)   │
                     │                                │
                     │ • Next Token Prediction        │
                     │ • Masked Language Modeling     │
                     │ • Huge datasets                │
                     │ • Large GPU clusters           │
                     └─────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │  Base Pretrained Model (LLM)    │
                    │  Example: LLaMA, GPT, Mistral   │
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │   2. Supervised Fine-Tuning     │
                    │            (SFT)                │
                    │                                 │
                    │ Instruction / QA Datasets       │
                    │                                 │
                    │ Example:                       │
                    │ - Question → Answer             │
                    │ - Instruction → Response        │
                    │                                 │
                    │ Techniques:                     │
                    │ • Full Fine-tuning              │
                    │ • PEFT (LoRA / QLoRA / etc)     │
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │      Instruction-Tuned Model     │
                   │       (Assistant behavior)       │
                   └─────────────┬────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────────────┐
                   │        3. Preference Alignment   │
                   │            (RLHF / DPO)          │
                   │                                  │
                   │ Human Feedback                   │
                   │ Response Ranking                 │
                   │ Safety Alignment                 │
                   └─────────────┬────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────────────┐
                   │          Final LLM               │
                   │   Chatbot / Assistant / API     │
                   │                                  │
                   │ Examples:                        │
                   │ - ChatGPT-like models            │
                   │ - Domain-specific assistants     │
                   └──────────────────────────────────┘
```

------

[Explanation DOC](https://youtu.be/CcrC5zSv1iA?t=911)

