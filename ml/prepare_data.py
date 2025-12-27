"""
Data Preparation Script for Activity-Field Mapping
Loads JSONL data, formats it for fine-tuning, and splits into train/val sets
"""

import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import random


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_data_for_training(data: List[Dict]) -> List[str]:
    """
    Format data for GPT-style training
    Format: Activity: [prompt] → Field: [completion]
    """
    formatted_data = []
    for item in data:
        prompt = item['prompt'].strip()
        completion = item['completion'].strip()
        
        # Create training text in a conversational format
        text = f"Activity: {prompt} Field: {completion}"
        formatted_data.append(text)
    
    return formatted_data


def split_data(data: List[str], train_split: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split data into train and validation sets"""
    train_data, val_data = train_test_split(
        data, 
        train_size=train_split, 
        random_state=seed,
        shuffle=True
    )
    return train_data, val_data


def save_dataset(data: List[str], output_path: str):
    """Save formatted data to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to prepare and split data"""
    print("=" * 60)
    print("Activity-Field Mapping - Data Preparation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    data_path = config['data']['data_path']
    train_split = config['data']['train_split']
    seed = config['data']['seed']
    
    print(f"\n📂 Loading data from: {data_path}")
    
    # Load JSONL data
    raw_data = load_jsonl(data_path)
    print(f"✅ Loaded {len(raw_data)} examples")
    
    # Format data
    print(f"\n🔄 Formatting data for training...")
    formatted_data = format_data_for_training(raw_data)
    
    # Display sample
    print(f"\n📝 Sample formatted data:")
    for i, example in enumerate(formatted_data[:3], 1):
        print(f"  {i}. {example[:100]}..." if len(example) > 100 else f"  {i}. {example}")
    
    # Split data
    print(f"\n✂️  Splitting data (train: {train_split*100}%, val: {(1-train_split)*100}%)")
    train_data, val_data = split_data(formatted_data, train_split, seed)
    print(f"   Training examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    save_dataset(train_data, "data/train.json")
    save_dataset(val_data, "data/val.json")
    print(f"   Saved to: data/train.json and data/val.json")
    
    print(f"\n✅ Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
