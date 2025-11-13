#!/usr/bin/env python3
"""
Quick test script to verify local setup.
Run this after training the model and before deploying.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import tokenizers
        import fastapi
        import pandas
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_artifacts():
    """Test that model artifacts exist."""
    print("\nTesting artifacts...")
    artifacts_dir = Path("artifacts")
    model_path = artifacts_dir / "model.pt"
    tokenizer_path = artifacts_dir / "tokenizer.json"
    
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        print("  Run: python -m ml_model.train --train-tokenizer")
        return False
    print(f"✓ Model found: {model_path}")
    
    if not tokenizer_path.exists():
        print(f"✗ Tokenizer not found at {tokenizer_path}")
        print("  Run: python -m ml_model.train --train-tokenizer")
        return False
    print(f"✓ Tokenizer found: {tokenizer_path}")
    
    return True

def test_ml_model():
    """Test that ml_model package can be imported."""
    print("\nTesting ml_model package...")
    try:
        from ml_model import inference, model
        print("✓ ml_model package imports successfully")
        return True
    except ImportError as e:
        print(f"✗ ml_model import error: {e}")
        print("  Run: pip install -e .")
        return False

def test_backend():
    """Test that backend can be imported."""
    print("\nTesting backend...")
    try:
        sys.path.insert(0, str(Path("backend").absolute()))
        from app import main
        print("✓ Backend imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Backend import error: {e}")
        return False

def main():
    print("=" * 50)
    print("Local Setup Test")
    print("=" * 50)
    
    results = [
        test_imports(),
        test_artifacts(),
        test_ml_model(),
        test_backend(),
    ]
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed! Ready for deployment.")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

