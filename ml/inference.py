"""
Inference Script for Activity-Field Prediction
Loads trained model and provides prediction interface
"""

import yaml
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
from typing import Dict, Optional
import argparse


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ActivityFieldPredictor:
    """Class for predicting fields from activities"""
    
    def __init__(self, model_path: str, config: dict):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model directory
            config: Configuration dictionary
        """
        print(f"📥 Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        try:
            # Try loading as PEFT model first
            self.model = GPT2LMHeadModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️  Could not load as standard model: {e}")
            print(f"   Trying to load as PEFT model...")
            base_model_name = config['model']['base_model']
            base_model = GPT2LMHeadModel.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        
        self.model.eval()
        
        # Inference parameters
        self.temperature = config['inference']['temperature']
        self.max_new_tokens = config['inference']['max_new_tokens']
        self.top_p = config['inference']['top_p']
        self.do_sample = config['inference']['do_sample']
        
        print(f"✅ Model loaded successfully!")
    
    def predict(self, activity: str) -> Dict[str, str]:
        """
        Predict field for given activity
        
        Args:
            activity: Activity description
            
        Returns:
            Dictionary with 'field' and 'reason'
        """
        # Format input
        prompt = f"Activity: {activity} Field:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract field from generated text
        # Format is: "Activity: {activity} Field: {field} (reason: {reason})"
        try:
            # Remove the prompt part
            field_part = generated_text[len(prompt):].strip()
            
            # Parse field and reason
            if "(reason:" in field_part.lower():
                field = field_part.split("(reason:")[0].strip()
                reason = field_part.split("(reason:")[1].strip().rstrip(")")
            else:
                field = field_part
                reason = "No specific reason provided"
            
            return {
                "activity": activity,
                "field": field,
                "reason": reason,
                "full_response": field_part
            }
        except Exception as e:
            return {
                "activity": activity,
                "field": "Error parsing response",
                "reason": str(e),
                "full_response": generated_text
            }
    
    def batch_predict(self, activities: list) -> list:
        """Predict fields for multiple activities"""
        return [self.predict(activity) for activity in activities]


def test_model():
    """Test the model with sample activities"""
    config = load_config()
    model_path = config['model']['model_save_path']
    
    # Initialize predictor
    predictor = ActivityFieldPredictor(model_path, config)
    
    # Test activities
    test_activities = [
        "Design 4-wheel rover chassis →",
        "PID controller tuning for arm →",
        "Navier-Stokes equation solver →",
        "Convolutional Neural Network (CNN) →",
        "Solar panel efficiency optimization →",
        "Autonomous drone path planning →"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Model Predictions")
    print("=" * 60)
    
    for activity in test_activities:
        result = predictor.predict(activity)
        print(f"\n📝 Activity: {activity}")
        print(f"🎯 Field: {result['field']}")
        print(f"💡 Reason: {result['reason']}")
        print(f"   Full: {result['full_response']}")
    
    print("\n" + "=" * 60)


def interactive_mode():
    """Interactive CLI for testing predictions"""
    config = load_config()
    model_path = config['model']['model_save_path']
    
    # Initialize predictor
    predictor = ActivityFieldPredictor(model_path, config)
    
    print("\n" + "=" * 60)
    print("Interactive Activity-Field Prediction")
    print("=" * 60)
    print("Enter activities to get field predictions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        activity = input("🔍 Enter activity: ").strip()
        
        if activity.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not activity:
            continue
        
        result = predictor.predict(activity)
        print(f"🎯 Field: {result['field']}")
        print(f"💡 Reason: {result['reason']}\n")


def main():
    """Entry point with CLI arguments"""
    parser = argparse.ArgumentParser(description="Activity-Field Prediction Inference")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test predictions with sample activities"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--activity",
        type=str,
        help="Single activity to predict"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_model()
    elif args.interactive:
        interactive_mode()
    elif args.activity:
        config = load_config()
        predictor = ActivityFieldPredictor(config['model']['model_save_path'], config)
        result = predictor.predict(args.activity)
        print(f"\nActivity: {result['activity']}")
        print(f"Field: {result['field']}")
        print(f"Reason: {result['reason']}")
    else:
        print("Please specify --test, --interactive, or --activity")


if __name__ == "__main__":
    main()
