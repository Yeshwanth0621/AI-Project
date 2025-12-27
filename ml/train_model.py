"""
Model Training Script for Activity-Field Mapping
Fine-tunes DistilGPT-2 using LoRA for parameter-efficient training
"""

import json
import yaml
import torch
from pathlib import Path
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_from_json(file_path: str) -> List[str]:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataset(data: List[str], tokenizer, max_length: int = 128) -> Dataset:
    """Tokenize and prepare dataset for training"""
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({'text': data})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(config: dict):
    """Initialize model and tokenizer with LoRA"""
    model_name = config['model']['base_model']
    
    print(f"📥 Loading tokenizer: {model_name}")
    print(f"   Note: This requires internet connection to Hugging Face")
    
    # Set environment variable to use offline mode if models are cached
    # os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n❌ ERROR: Could not download tokenizer from Hugging Face")
        print(f"   Error details: {str(e)}")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Check your internet connection")
        print(f"   2. Try again (Hugging Face might be temporarily down)")
        print(f"   3. If behind a proxy, configure it:")
        print(f"      export HTTP_PROXY=http://your-proxy:port")
        print(f"      export HTTPS_PROXY=http://your-proxy:port")
        print(f"   4. Download model manually:")
        print(f"      git clone https://huggingface.co/distilgpt2")
        print(f"      Then update config.yaml base_model to local path")
        raise
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📥 Loading model: {model_name}")
    try:
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n❌ ERROR: Could not download model from Hugging Face")
        print(f"   Error details: {str(e)}")
        print(f"\n💡 See solutions above for tokenizer download issue")
        raise
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    print(f"🔧 Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(config: dict):
    """Main training function"""
    print("=" * 60)
    print("Activity-Field Mapping - Model Training")
    print("=" * 60)
    
    # Load datasets
    print(f"\n📂 Loading train and validation datasets...")
    train_data = load_dataset_from_json("data/train.json")
    val_data = load_dataset_from_json("data/val.json")
    print(f"   Train examples: {len(train_data)}")
    print(f"   Val examples: {len(val_data)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare datasets
    print(f"\n🔄 Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, config['model']['max_length'])
    val_dataset = prepare_dataset(val_data, tokenizer, config['model']['max_length'])
    
    # Training arguments
    output_dir = config['model']['model_save_path']
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        fp16=False,  # Set to True if using GPU
        report_to="none",
        save_total_limit=2,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print()
    
    train_result = trainer.train()
    
    # Save model
    print(f"\n💾 Saving final model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    
    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Training time: {metrics.get('train_runtime', 0):.2f}s")
    print("=" * 60)


def main():
    """Entry point"""
    config = load_config()
    
    print("\n🌐 Checking internet connection to Hugging Face...")
    print("   This is required to download the base model (distilgpt2)")
    print("   Download size: ~250 MB")
    print("")
    
    try:
        train(config)
    except Exception as e:
        print(f"\n\n❌ Training failed!")
        print(f"   Error: {str(e)}")
        print(f"\n📝 Troubleshooting tips:")
        print(f"   1. Ensure you have a stable internet connection")
        print(f"   2. Check if you can access: https://huggingface.co")
        print(f"   3. Try running again (downloads are cached after first success)")
        print(f"   4. Check firewall/antivirus settings")
        exit(1)


if __name__ == "__main__":
    main()
