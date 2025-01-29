# 🚀 LLaMA 3.1 Price Prediction Model

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-orange)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A fine-tuned LLaMA 3.1 model for predicting product prices based on descriptions
</div>

## 📝 Table of Contents

- [Project Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Results & Performance](#-results--performance)
- [Implementation Details](#-implementation-details)
- [Usage Guide](#-usage-guide)

## 🎯 Project Overview

This project implements a price prediction system using Meta's LLaMA 3.1 8B model, fine-tuned on product descriptions from the Amazon Appliances dataset. The model learns to predict product prices by understanding the relationship between product descriptions and their market values.

### Key Features

- 🤖 Fine-tuned LLaMA 3.1 model using LoRA
- 📊 TPU-accelerated training
- 💡 Intelligent price prediction
- 📈 Weights & Biases integration for monitoring
- 🔍 Advanced inference with weighted predictions

<img src="_asserts/here we can see the llamma model rchetecture q_proj layer is replaced by the lora_q_proj.png" alt="LoRA Architecture" width="800"/>

> The architecture shows how LoRA adapts the LLaMA model by replacing the query projection layer (q_proj) with a low-rank adaptation layer (lora_q_proj)

## 🏗 Model Architecture

### Base Model
- **Model**: Meta-LLaMA-3.1-8B
- **Type**: Causal Language Model
- **Parameters**: 8 Billion
- **Quantization**: 4-bit precision

### LoRA Configuration
```python
lora_parameters = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

<img src="_asserts/here in the infrencing when we have loaded the model its size is increased.png" alt="Model Size" width="800"/>

> During inference, the model size increases due to the loaded weights and LoRA adaptations

## 📈 Training Process

### Dataset Preparation
- Source: Amazon-Reviews-2023 (Appliances category)
- Split: 25,000 training samples, 2,000 test samples
- Features: Product descriptions, titles, features, and prices

### Training Configuration
```python
training_args = SFTConfig(
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine'
)
```

### Training Progress

<img src="_asserts/training progress of llama.png" alt="Training Progress" width="800"/>

> The training progress shows the model's learning curve across epochs

<img src="_asserts/train epoch vs learning rate loss.png" alt="Learning Rate vs Loss" width="800"/>

> Correlation between learning rate adjustments and training loss

<img src="_asserts/weights and bias traning loss chart.png" alt="W&B Loss Chart" width="800"/>

> Weights & Biases monitoring showing the training loss over time

### Advanced Metrics

<img src="_asserts/train loss graph of wegihts and bias.png" alt="Detailed Loss Analysis" width="800"/>

> Detailed analysis of weights and biases during training

<img src="_asserts/training progress on 3550 steps.png" alt="Training Steps" width="800"/>

> Progress visualization after 3,550 training steps

## 🎯 Results & Performance

### Model Evaluation

<img src="_asserts/graph of the performance of the llamma model.png" alt="Model Performance" width="800"/>

> Overall performance metrics of the fine-tuned model

<img src="_asserts/results of the llamma fine tune.png" alt="Fine-tuning Results" width="800"/>

> Final results after fine-tuning, showing prediction accuracy

### Key Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Logarithmic Error (RMSLE)
- Price Prediction Accuracy within 20% threshold

## 💻 Implementation Details

### Data Processing
```python
def extract_price(s):
    if "Price is $" in s:
        contents = s.split("Price is $")[1]
        contents = contents.replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
    return 0
```

### Inference Process
```python
def improved_model_predict(prompt, device="cuda"):
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, device=device)
    with torch.no_grad():
        outputs = fine_tuned_model(inputs, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :].to('cpu')
```

## 🚀 Usage Guide

1. **Environment Setup**
```bash
pip install -r requirements.txt
```

2. **Model Training**
```bash
python fine-tune-llama-3-1.py
```

3. **Price Prediction**
```python
from fine_tune_llama_3_1 import improved_model_predict

prediction = improved_model_predict("Your product description here")
print(f"Predicted Price: ${prediction:.2f}")
```

## 📊 Performance Monitoring

The project uses Weights & Biases for comprehensive training monitoring:
- Loss tracking
- Learning rate scheduling
- Model checkpointing
- Performance metrics visualization

---

<div align="center">
Made with ❤️ using LLaMA 3.1 and PyTorch
</div>
