import torch
from transformers import BitsAndBytesConfig

class Hyperparameters:
    # Model config
    MODEL_NAME = "google/gemma-3-4b-it"
    MAX_SEQ_LENGTH = 2048
    USE_FLASH_ATTENTION = True

    # NEW: Quantization Config
    USE_QUANTIZATION = True
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"  # Normalized Float 4
    BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
    BNB_4BIT_USE_DOUBLE_QUANT = True  # Double quantization for even more memory savings
    
    # LoRA config
    LORA_R = 64
    LORA_ALPHA = 16
    LORA_DROPOUT = 0
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training config
    SEED = 0
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8
    WARMUP_RATIO = 0.03
    MAX_GRAD_NORM = 0.3
    
    # Data config
    TRAIN_SAMPLES_PER_CLASS = 300
    TEST_SAMPLES_PER_CLASS = 300
    EVAL_SAMPLES_PER_CLASS = 50

    # Get BitsAndBytes Config
    @property
    def get_bnb_config(self):
        if not self.USE_QUANTIZATION:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=self.LOAD_IN_4BIT,
            bnb_4bit_quant_type=self.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, self.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=self.BNB_4BIT_USE_DOUBLE_QUANT,
        )

config = Hyperparameters()