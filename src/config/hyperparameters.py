class Hyperparameters:
    # Model config
    MODEL_NAME = "google/gemma-3-4b-it"
    MAX_SEQ_LENGTH = 2048
    USE_FLASH_ATTENTION = True
    
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

config = Hyperparameters()