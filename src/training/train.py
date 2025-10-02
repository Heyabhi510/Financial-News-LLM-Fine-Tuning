from transformers import TrainingArguments
from trl import SFTTrainer
from models.lora_config import get_lora_config
from config.hyperparameters import config
from config.paths import paths

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_training_args(self):
        """Your existing training arguments"""
        return TrainingArguments(
            output_dir=str(paths.TRAINED_MODEL),
            seed=config.SEED,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            learning_rate=config.LEARNING_RATE,
            warmup_ratio=config.WARMUP_RATIO,
            max_grad_norm=config.MAX_GRAD_NORM,
            logging_steps=25,
            eval_steps=112,
            evaluation_strategy="steps",
            save_strategy="no",
            report_to="tensorboard",
            bf16=True,
        )
    
    def create_trainer(self, train_dataset, eval_dataset):
        """Create SFTTrainer instance"""
        training_args = self.get_training_args()
        peft_config = get_lora_config()
        
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            tokenizer=self.tokenizer,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dataset_text_field="text",
        )
        
        return trainer