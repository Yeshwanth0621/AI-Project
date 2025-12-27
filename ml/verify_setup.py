"""
Quick verification script to check if all components are ready
Run this before training to ensure everything is set up correctly
"""

import sys
from pathlib import Path
import importlib.util


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False


def check_module(module_name: str) -> bool:
    """Check if a Python module is installed"""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ Module installed: {module_name}")
        return True
    else:
        print(f"❌ Module missing: {module_name}")
        return False


def main():
    print("=" * 60)
    print("🔍 SLM Training System - Pre-flight Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check data file
    print("📂 Checking Data Files:")
    all_ok &= check_file_exists("../data.jsonl", "Training data")
    print()
    
    # Check config
    print("⚙️  Checking Configuration:")
    all_ok &= check_file_exists("config.yaml", "Config file")
    print()
    
    # Check scripts
    print("📜 Checking Scripts:")
    all_ok &= check_file_exists("prepare_data.py", "Data preparation")
    all_ok &= check_file_exists("train_model.py", "Training script")
    all_ok &= check_file_exists("inference.py", "Inference script")
    print()
    
    # Check Python modules
    print("📦 Checking Python Dependencies:")
    required_modules = [
        "transformers",
        "torch",
        "peft",
        "datasets",
        "sklearn",
        "yaml",
        "fastapi",
        "uvicorn"
    ]
    
    for module in required_modules:
        module_ok = check_module(module)
        all_ok &= module_ok
        if not module_ok:
            print(f"   Install with: pip install {module}")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✅ All checks passed! You're ready to train.")
        print()
        print("Next steps:")
        print("  1. python prepare_data.py")
        print("  2. python train_model.py")
        print("  3. python inference.py --test")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("To install all dependencies:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
