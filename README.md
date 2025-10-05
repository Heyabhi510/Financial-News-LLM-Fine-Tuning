# 🏦 Financial News Sentiment Analysis with Gemma-3-4B

> Fine-tuning Google's Gemma-3-4B model for financial sentiment classification using **QLoRA (4-bit Quantized LoRA)** - achieving **77% accuracy** with only **0.2% trainable parameters**.


## 📊 Project Overview

This project demonstrates **end-to-end fine-tuning** of a large language model (Gemma-3B) for financial domain adaptation. The model is trained to analyze sentiment in financial news articles and classify them as **Positive, Negative, or Neutral**.


### 🎯 Key Features

- **🔧 Hugging Face Transformers and Parameter Efficient Fine-Tuning** using QLoRA (4-bit Quantized Low-Rank Adaptation)
- **📈 Financial Domain Adaptation** of pre-trained LLM
- **🔄 Modular & Scalable Architecture** with proper code organization
- **📊 Comprehensive Evaluation** with accuracy, F1-score, and confusion matrices


## 📁 Project Structure
FINANCIAL NEWS LLM FINE-TUNING
├── src/
│ ├── config/ # Configuration management
│ │ ├── hyperparameters.py # Training parameters
│ │ └── paths.py # File paths
│ ├── data/ # Data processing
│ │ ├── loader.py # Data loading & validation
│ │ ├── preprocessor.py # Prompt engineering
│ │ └── splitter.py # Train/test splitting
│ ├── models/ # Model architecture
│ │ ├── setup.py # Model & tokenizer initialization
│ │ └── lora_config.py # LoRA configuration
│ ├── training/ # Training pipeline
│ │ └── train.py # Training configuration
│ ├── evaluation/ # Model evaluation
│ │ ├── metrics.py # Performance metrics
│ │ └── predictor.py # Inference functions
│ ├── scripts/ # Execution scripts
│ │ └── train_model.py # Main training script
│ └── outputs/ # Generated artifacts
│ └── predictions/ # Prediction results
│ └── trained_models/# Saved model weights
├── requirements.txt # Dependencies
└── README.md


# 🚀 How to Run
## Clone repository
git clone https://github.com/Heyabhi510/Financial-News-LLM-Fine-Tuning.git
cd Financial-News-LLM-Fine-Tuning

## Install dependencies
pip install -r requirements.txt

## Setup Hugging Face token
export HF_Token="your_huggingface_token"

## Run the complete training pipeline
python src/scripts/train_model.py


# 📦 Requirements
- torch>=2.0.0
- transformers>=4.35.0
- datasets>=2.14.0
- accelerate>=0.24.0
- peft>=0.7.0
- trl>=0.7.0
- bitsandbytes>=0.41.0
- pandas>=1.5.0
- scikit-learn>=1.2.0
- tqdm>=4.65.0
- matplotlib>=3.7.0
- tensorboard>=2.13.0
- flash-attn>=2.0.0


# 🚀 Performance Optimizations
## Memory Efficiency
- 4-bit QLoRA: Reduces memory usage by 60%
- Gradient Checkpointing: Trading compute for memory
- Flash Attention 2: Faster attention computation
- BF16 Precision: Better numerical stability

## Training Speed
- Paged AdamW 8-bit: Memory-efficient optimizer
- Gradient Accumulation: Effective larger batches
- Mixed Precision: BF16 training


# 🔧 Technical Details
## Model Architecture
- Base Model: Google Gemma-4B
- Fine-tuning Method: 4-bit QLoRA
- Adapter Rank: 64
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Configuration
- Batch Size: 8 (with gradient accumulation)
- Learning Rate: 2e-4 (with cosine scheduler)
- Epochs: 5
- Max Sequence Length: 2048 tokens
- Warmup Ratio: 0.03

## Dataset
- Source: Financial PhraseBank
- Size: 5,000 labeled samples
- Classes: Positive, Negative, Neutral
- Train/Val/Test Split per class: 300/50/300


# 📈 Results
## Performance Metrics
Model	                Accuracy	F1-Score	Precision	Recall
Baseline (Zero-shot)	45.2%	    0.43	    0.41	    0.45
Fine-tuned Gemma-4B	    78.6%	    0.77	    0.79	    0.76


# 👨‍💻 Author
## Your Name
- GitHub: <a href='https://github.com/Heyabhi510'>Heyabhi510</a>
- LinkedIn: <a href='www.linkedin.com/in/abhi-s-thakkar'>Abhishek Thakkar</a>


# 🙏 Acknowledgments
- HuggingFace for the <a href='https://github.com/huggingface/transformers'>Transformers</a> and <a href='https://github.com/huggingface/peft'>PEFT</a> libraries
- Google for releasing <a href='https://huggingface.co/google/gemma-3-4b-it'>Gemma models</a>
- <a href='https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10'>Financial PhraseBank</a> dataset creators
