import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.hyperparameters import config

class ModelSetup:
    def __init__(self, hf_token):
        self.hf_token = hf_token
    
    def setup_model(self):
        # Use the faster "flash_attention_2" if installed, otherwise fall back to the eager implementation.
        attn_implementation = "flash_attention_2" if config.USE_FLASH_ATTENTION else "eager"
        
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            dtype="auto",
            device_map="auto",
            attn_implementation=attn_implementation,
            token=self.hf_token
        )
        
        return model
    
    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, 
            max_length=config.MAX_SEQ_LENGTH
        )
        
        # Tokenizer configuration
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer