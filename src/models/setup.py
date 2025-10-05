import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config.hyperparameters import config

class ModelSetup:
    def __init__(self, hf_token):
        self.hf_token = hf_token
    
    def setup_model(self):
        # Use the faster "flash_attention_2" if installed, otherwise fall back to the eager implementation.
        attn_implementation = "flash_attention_2" if config.USE_FLASH_ATTENTION else "eager"

        # Get quantization config
        bnb_config = config.get_bnb_config

        if bnb_config:
            print("🔢 Using 4-bit quantization with bitsandbytes")
            print(f"   - Quant Type: {config.BNB_4BIT_QUANT_TYPE}")
            print(f"   - Compute Dtype: {config.BNB_4BIT_COMPUTE_DTYPE}")
            print(f"   - Double Quant: {config.BNB_4BIT_USE_DOUBLE_QUANT}")
        else:
            print("⚠️  Running without quantization")
        

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                dtype="auto",
                device_map="auto",
                attn_implementation=attn_implementation,
                token=self.hf_token
            )

             # Enable gradient checkpointing for memory efficiency
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
        
            return model
        
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Fallback: try without quantization
            print("Trying without quantization...")
            return AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=self.hf_token
            )
    
    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, 
            max_length=config.MAX_SEQ_LENGTH
        )
        
        # Tokenizer configuration
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer