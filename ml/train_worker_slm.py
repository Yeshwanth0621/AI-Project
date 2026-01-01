"""
SLM Training Script for Worker Recommendation Reasoning
Fine-tunes DistilGPT-2 to generate explanations for worker recommendations
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_config():
    """Load configuration"""
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_training_data():
    """Load worker recommendation training data"""
    print("📂 Loading training data...")
    
    with open("data/worker_train.json", "r") as f:
        train_data = json.load(f)
    
    with open("data/worker_val.json", "r") as f:
        val_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data)} train and {len(val_data)} val examples")
    
    return train_data, val_data


def prepare_dataset(data, tokenizer, max_length=128):
    """Tokenize and prepare dataset"""
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    dataset = Dataset.from_dict({'text': data})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def main():
    print("=" * 60)
    print("Worker Recommendation SLM Training")
    print("=" * 60)
    
    # Load config
    config = load_config()
    model_path = config['model']['base_model']
    
    # Load data
    train_data, val_data = load_training_data()
    
    # Load model and tokenizer
    print(f"\n📥 Loading model: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Apply LoRA
    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    print("\n🔄 Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, config['model']['max_length'])
    val_dataset = prepare_dataset(val_data, tokenizer, config['model']['max_length'])
    
    # Training arguments
    output_dir = "worker_model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # Fewer epochs since we have good data
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
        report_to="none",
        save_total_limit=2,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Epochs: 10")
    print(f"   Batch size: 4")
    print()
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ SLM training complete!")
    print("=" * 60)
    print("\nNext: Build hybrid recommender with: python hybrid_recommender.py")


if __name__ == "__main__":
    main()
