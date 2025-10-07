from peft import LoraConfig
from config.hyperparameters import config

def get_lora_config():
    """LoRA configuration"""
    return LoraConfig(
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        r=config.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES
    )