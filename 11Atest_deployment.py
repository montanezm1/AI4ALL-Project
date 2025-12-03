"""
TEST SCRIPT - Verify Models Load Correctly
===========================================
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
required_files = [
    'michael_model.pth',
    'best_osteoporosis_model.pth',
    'sydney_best_model.pth'
]

files_found = []
files_missing = []

for file in required_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   ✓ {file} ({size_mb:.1f} MB)")
        files_found.append(file)
    else:
        print(f"   ✗ {file} NOT FOUND")
        files_missing.append(file)

if files_missing:
    print(f"\n   ⚠️  WARNING: {len(files_missing)} model file(s) missing!")
    print(f"   Missing: {', '.join(files_missing)}")
    print("\n   The dashboard will still run but some models won't be available.")
else:
    print(f"\n   ✓ All {len(required_files)} model files found!")

# ========== 3. DEFINE MODEL ARCHITECTURES ==========
print("\n3. Testing Model Architectures...")

class MichaelModel(nn.Module):
    """ResNet50"""
    def __init__(self, num_classes=3):
        super(MichaelModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class DiaModel(nn.Module):
    """EfficientNet-B3"""
    def __init__(self, num_classes=3):
        super(DiaModel, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=False)
        in_features = 1536
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SydneyModel(nn.Module):
    """ResNet18"""
    def __init__(self, num_classes=3):
        super(SydneyModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

print("   ✓ Model architectures defined")

# ========== 4. LOAD MODELS ==========
print("\n4. Loading Models...")
device = torch.device('cpu')  # Use CPU for testing
models_loaded = {}

# Michael's Model
if 'michael_model.pth' in files_found:
    try:
        model = MichaelModel(num_classes=3).to(device)
        model.load_state_dict(torch.load('michael_model.pth', map_location=device))
        model.eval()
        models_loaded['Michael'] = model
        print("   ✓ Michael's ResNet50 loaded successfully")
    except Exception as e:
        print(f"   ✗ Michael's model failed: {e}")

# Dia's Model
if 'best_osteoporosis_model.pth' in files_found:
    try:
        model = DiaModel(num_classes=3).to(device)
        model.load_state_dict(torch.load('best_osteoporosis_model.pth', map_location=device))
        model.eval()
        models_loaded['Dia'] = model
        print("   ✓ Dia's EfficientNet-B3 loaded successfully")
    except Exception as e:
        print(f"   ✗ Dia's model failed: {e}")

# Sydney's Model
if 'sydney_best_model.pth' in files_found:
    try:
        model = SydneyModel(num_classes=3).to(device)
        checkpoint = torch.load('sydney_best_model.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        models_loaded['Sydney'] = model
        print("   ✓ Sydney's ResNet18 loaded successfully")
    except Exception as e:
        print(f"   ✗ Sydney's model failed: {e}")

# ========== 5. TEST INFERENCE ==========
print("\n5. Testing Inference...")
if models_loaded:
    try:
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        for model_name, model in models_loaded.items():
            output = model(dummy_input)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item() * 100
            
            print(f"   ✓ {model_name}: Output shape {output.shape}, Prediction class {predicted_class}, Confidence {confidence:.1f}%")
        
        print("\n   ✓ All models can perform inference!")
    except Exception as e:
        print(f"   ✗ Inference test failed: {e}")
else:
    print("   ⚠️  No models loaded, skipping inference test")

# ========== 6. CHECK DEPENDENCIES ==========
print("\n6. Checking Required Dependencies...")
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
    print(f"\n   ⚠️  Install missing packages:")
    print(f"   pip install {' '.join(missing_deps)}")

# ========== 7. FINAL SUMMARY ==========
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_checks = 7
passed_checks = 0

if sys.version_info >= (3, 8):
    passed_checks += 1
    print("✓ Python version OK")
else:
    print("✗ Python version too old (need 3.8+)")

if not files_missing:
    passed_checks += 1
    print("✓ All model files present")
else:
    print(f"✗ {len(files_missing)} model file(s) missing")

if len(models_loaded) == 3:
    passed_checks += 1
    print("✓ All 3 models loaded successfully")
elif models_loaded:
    print(f"⚠️  Only {len(models_loaded)}/3 models loaded")
else:
    print("✗ No models loaded")

if models_loaded:
    passed_checks += 1
    print("✓ Inference test passed")
else:
    print("✗ Could not test inference")

if not missing_deps:
    passed_checks += 1
    print("✓ All dependencies installed")
else:
    print(f"✗ {len(missing_deps)} dependencies missing")

print("\n" + "="*60)
if passed_checks >= 4:
    print("🎉 READY TO DEPLOY!")
    print("\nRun the dashboard with:")
    print("    streamlit run osteoporosis_dashboard.py")
else:
    print("⚠️  FIX ISSUES BEFORE DEPLOYING")
    print("\nAddress the errors above, then run this test again.")

print("="*60)
