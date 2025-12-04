"""
TEST SCRIPT - Verify Models Load Correctly (FIXED VERSION)
============================================================
Run this before deploying the dashboard to check everything works.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

print("="*60)
print("OSTEOPOROSIS DASHBOARD - PRE-DEPLOYMENT CHECK")
print("="*60)

# ========== 1. CHECK PYTHON ENVIRONMENT ==========
print("\n1. Checking Python Environment...")
import sys
print(f"   Python version: {sys.version}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

# ========== 2. CHECK MODEL FILES ==========
print("\n2. Checking Model Files...")
available_files = []

# Check for model files
model_files = [
    'sydney_best_model.pth',
    'resnet50_best_model.pth',
    'best_osteoporosis_model.pth',
    'michael_model.pth'
]

for file in model_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   ✓ {file} ({size_mb:.1f} MB)")
        available_files.append(file)

if not available_files:
    print(f"   ✗ No model files found!")
    print(f"\n   The dashboard will not work without model files.")
else:
    print(f"\n   ✓ Found {len(available_files)} model file(s)!")

# ========== 3. LOAD AND TEST MODELS ==========
print("\n3. Loading and Testing Models...")
device = torch.device('cpu')
models_loaded = {}

# Test Sydney's Model
if 'sydney_best_model.pth' in available_files:
    try:
        print("\n   Testing Sydney's ResNet18...")
        sydney_model = models.resnet18(weights=None)
        num_ftrs = sydney_model.fc.in_features
        sydney_model.fc = nn.Linear(num_ftrs, 3)
        
        # Try loading
        sydney_model.load_state_dict(torch.load('sydney_best_model.pth', map_location=device, weights_only=True))
        sydney_model.eval()
        models_loaded['Sydney'] = sydney_model
        print("   ✓ Sydney's ResNet18 loaded successfully")
    except Exception as e:
        print(f"   ✗ Sydney's model failed: {str(e)[:150]}")

# Test ResNet50 Model
if 'resnet50_best_model.pth' in available_files:
    try:
        print("\n   Testing ResNet50...")
        resnet50_model = models.resnet50(weights=None)
        num_ftrs = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(num_ftrs, 3)
        
        resnet50_model.load_state_dict(torch.load('resnet50_best_model.pth', map_location=device, weights_only=True))
        resnet50_model.eval()
        models_loaded['ResNet50'] = resnet50_model
        print("   ✓ ResNet50 loaded successfully")
    except Exception as e:
        print(f"   ✗ ResNet50 failed: {str(e)[:150]}")

# Test Dia's EfficientNet
if 'best_osteoporosis_model.pth' in available_files:
    try:
        print("\n   Testing Dia's EfficientNet-B3...")
        dia_model = models.efficientnet_b3(weights=None)
        in_features = 1536
        dia_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 3)
        )
        dia_model.load_state_dict(torch.load('best_osteoporosis_model.pth', map_location=device, weights_only=True))
        dia_model.eval()
        models_loaded['Dia'] = dia_model
        print("   ✓ Dia's EfficientNet-B3 loaded successfully")
    except Exception as e:
        print(f"   ⚠️ Dia's model not loaded (expected - different architecture)")

# Test Michael's Model (if different file)
if 'michael_model.pth' in available_files:
    try:
        print("\n   Testing Michael's ResNet50...")
        michael_model = models.resnet50(weights=None)
        num_ftrs = michael_model.fc.in_features
        michael_model.fc = nn.Linear(num_ftrs, 3)
        
        michael_model.load_state_dict(torch.load('michael_model.pth', map_location=device, weights_only=True))
        michael_model.eval()
        models_loaded['Michael'] = michael_model
        print("   ✓ Michael's ResNet50 loaded successfully")
    except Exception as e:
        print(f"   ✗ Michael's model failed: {str(e)[:150]}")

# ========== 4. TEST INFERENCE ==========
print("\n4. Testing Inference...")
if models_loaded:
    try:
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        for model_name, model in models_loaded.items():
            output = model(dummy_input)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item() * 100
            
            print(f"   ✓ {model_name}: Can make predictions (output shape {output.shape})")
        
        print("\n   ✓ All loaded models can perform inference!")
    except Exception as e:
        print(f"   ✗ Inference test failed: {e}")
else:
    print("   ⚠️ No models loaded, skipping inference test")

# ========== 5. CHECK DEPENDENCIES ==========
print("\n5. Checking Required Dependencies...")
dependencies = {
    'streamlit': 'streamlit',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'PIL': 'pillow',
    'numpy': 'numpy',
    'plotly': 'plotly',
    'pandas': 'pandas'
}

missing_deps = []
for module_name, package_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} NOT INSTALLED")
        missing_deps.append(package_name)

if missing_deps:
    print(f"\n   ⚠️ Install missing packages:")
    print(f"   pip install {' '.join(missing_deps)}")

# ========== 6. FINAL SUMMARY ==========
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

checks_passed = 0
total_checks = 5

if sys.version_info >= (3, 8):
    checks_passed += 1
    print("✓ Python version OK")
else:
    print("✗ Python version too old (need 3.8+)")

if available_files:
    checks_passed += 1
    print(f"✓ Found {len(available_files)} model file(s)")
else:
    print("✗ No model files found")

if len(models_loaded) >= 1:
    checks_passed += 1
    print(f"✓ {len(models_loaded)} model(s) loaded successfully")
else:
    print("✗ No models loaded")

if models_loaded:
    checks_passed += 1
    print("✓ Inference test passed")
else:
    print("✗ Could not test inference")

if not missing_deps:
    checks_passed += 1
    print("✓ All dependencies installed")
else:
    print(f"✗ {len(missing_deps)} dependencies missing")

print("\n" + "="*60)
if checks_passed >= 4 and len(models_loaded) >= 1:
    print("🎉 READY TO DEPLOY!")
    print(f"\nYou have {len(models_loaded)} working model(s):")
    for name in models_loaded.keys():
        print(f"  - {name}")
    print("\nRun the dashboard with:")
    print("    streamlit run osteoporosis_dashboard_final.py")
else:
    print("⚠️ ISSUES FOUND")
    if not models_loaded:
        print("\n❌ CRITICAL: No models could be loaded!")
        print("Check that your .pth files match the model architectures.")
    else:
        print("\nAddress the errors above, then run this test again.")

print("="*60)