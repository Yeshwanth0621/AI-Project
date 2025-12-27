"""
Fixed model download script for Python 3.13 compatibility
"""

from huggingface_hub import snapshot_download
import os

print("=" * 60)
print("Downloading DistilGPT-2 Model (Python 3.13 Compatible)")
print("=" * 60)
print()

# Set cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"Cache directory: {cache_dir}")
print()

try:
    print("Downloading distilgpt2 model files...")
    print("This may take a few minutes (~250 MB)")
    print()
    
    # Download entire model repository
    local_dir = snapshot_download(
        repo_id="distilgpt2",
        cache_dir=cache_dir,
        resume_download=True
    )
    
    print(f"\n✓ Model downloaded successfully!")
    print(f"  Location: {local_dir}")
    print()
    print("=" * 60)
    print("SUCCESS! Now testing if it loads...")
    print("=" * 60)
    print()
    
    # Test loading
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir, local_files_only=True)
    print("✓ Tokenizer loaded!")
    
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(local_dir, local_files_only=True)
    print("✓ Model loaded!")
    
    print()
    print("="  * 60)
    print("ALL TESTS PASSED! Ready for training")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
