from trl import SFTTrainer, SFTConfig
from models.lora_config import get_lora_config
from config.hyperparameters import config
from config.paths import paths

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_training_args(self):
        return SFTConfig(
            output_dir=str(paths.TRAINED_MODEL),
            seed=config.SEED,
            num_train_epochs=config.NUM_EPOCHS,
            gradient_checkpointing=True,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            optim="adamw_torch_fused",
            save_steps=0,
            logging_steps=25,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            fp16=True,
            bf16=False,
            max_grad_norm=config.MAX_GRAD_NORM,
            max_steps=-1,
            warmup_ratio=config.WARMUP_RATIO,
            group_by_length=False,
            eval_strategy="steps",
            eval_steps=112,
            eval_accumulation_steps=1,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            packing=False,
            max_length=config.MAX_SEQ_LENGTH,
            report_to="tensorboard"
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
            processing_class=self.tokenizer,
            peft_config=peft_config
        )
        
        return trainer