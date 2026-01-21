"""
Verify that the installation is correct and all dependencies are available.

This script checks:
1. Python version
2. PyTorch and CUDA availability
3. All required packages
4. GPU detection and memory
5. Directory structure

Run this immediately after installation to ensure reproducibility.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("=" * 60)
    print("CHECKING PYTHON VERSION")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 12:
        print("✓ Python 3.12 detected (recommended)")
        return True
    elif version.major == 3 and version.minor >= 10:
        print("⚠ Python 3.10+ detected (should work, but 3.12 recommended)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} not supported")
        print("  Please use Python 3.10 or higher (3.12 recommended)")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA."""
    print("\n" + "=" * 60)
    print("CHECKING PYTORCH & CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: True")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            
            # GPU info
            gpu_count = torch.cuda.device_count()
            print(f"  GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # Test GPU
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor @ test_tensor.t()
                print("✓ GPU computation test passed")
                del test_tensor, result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ GPU computation test failed: {e}")
                return False
                
        else:
            print("⚠ CUDA not available (CPU-only mode)")
            print("  GPU acceleration disabled - training will be very slow")
        
        return True
        
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return False

def check_required_packages():
    """Check all required packages."""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED PACKAGES")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'cv2': 'OpenCV (opencv-python)',
        'skimage': 'scikit-image',
        'PIL': 'Pillow',
        'h5py': 'h5py',
        'yaml': 'PyYAML',
        'einops': 'einops',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'tensorboard': 'tensorboard',
        'ultralytics': 'Ultralytics YOLOv8',
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable-Baselines3',
        'tqdm': 'tqdm',
    }
    
    all_installed = True
    
    for module_name, package_name in required_packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name:30s} version: {version}")
        except ImportError:
            print(f"✗ {package_name:30s} NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_directory_structure():
    """Check that essential directories exist."""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        'src/phase1',
        'src/phase5',
        'scripts',
        'training',
        'tests',
        'models',
        'outputs',
        'docs',
    ]
    
    optional_dirs = [
        'data/cure-tsd',
        'data/yolo_dataset_full',
        'data/masks',
        'runs/train',
        'runs/rl_training',
    ]
    
    all_exist = True
    
    print("\nRequired directories:")
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} MISSING")
            all_exist = False
    
    print("\nOptional directories (needed for experiments):")
    for dir_path in optional_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"⚠ {dir_path} (will be created when needed)")
    
    return all_exist

def check_model_files():
    """Check for pre-trained model files."""
    print("\n" + "=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    model_files = {
        'models/yolov8n.pt': 'YOLOv8n pretrained weights',
        'models/yolo11n.pt': 'YOLO11n pretrained weights',
        'runs/train/weights/best.pt': 'Fine-tuned YOLO model',
        'runs/rl_training/best_model.pth': 'Trained RL agent',
    }
    
    for file_path, description in model_files.items():
        full_path = base_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✓ {description:35s} ({size_mb:.2f} MB)")
        else:
            print(f"⚠ {description:35s} (not found - download needed)")

def check_dataset():
    """Check dataset availability."""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / 'data' / 'yolo_dataset_full'
    
    if not dataset_path.exists():
        print("⚠ Dataset not found at data/yolo_dataset_full")
        print("  Run: python scripts/download_dataset.py")
        print("  Or: python scripts/generate_full_dataset.py")
        return False
    
    # Check images and labels
    images_path = dataset_path / 'images'
    labels_path = dataset_path / 'labels'
    
    if images_path.exists() and labels_path.exists():
        # Count files
        train_images = list((images_path / 'train').glob('*.jpg')) if (images_path / 'train').exists() else []
        val_images = list((images_path / 'val').glob('*.jpg')) if (images_path / 'val').exists() else []
        
        print(f"✓ Dataset directory exists")
        print(f"  Training images: {len(train_images)}")
        print(f"  Validation images: {len(val_images)}")
        print(f"  Total: {len(train_images) + len(val_images)}")
        
        # Expected: 28,727 images total
        total = len(train_images) + len(val_images)
        expected = 28727
        
        if abs(total - expected) < 100:  # Allow small variance
            print(f"✓ Dataset size matches expected (~{expected} images)")
            return True
        else:
            print(f"⚠ Dataset size mismatch (expected ~{expected}, found {total})")
            return False
    else:
        print("✗ Dataset structure incomplete")
        return False

def print_summary(results):
    """Print summary of checks."""
    print("\n" + "=" * 60)
    print("INSTALLATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Python Version", results['python']),
        ("PyTorch & CUDA", results['pytorch']),
        ("Required Packages", results['packages']),
        ("Directory Structure", results['directories']),
    ]
    
    all_passed = all(result for _, result in checks)
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:25s} {status}")
    
    print("\nOptional components:")
    print(f"  Model files: {'✓ Available' if results.get('models') else '⚠ Download needed'}")
    print(f"  Dataset: {'✓ Available' if results.get('dataset') else '⚠ Download needed'}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ INSTALLATION VERIFIED - Ready to reproduce experiments!")
        print("\nNext steps:")
        print("1. Download dataset: python scripts/download_dataset.py")
        print("2. Download models: python scripts/download_models.py")
        print("3. Run experiments: python scripts/reproduce_all.py")
    else:
        print("✗ INSTALLATION INCOMPLETE - Please fix issues above")
        print("\nRefer to REPRODUCIBILITY.md for detailed instructions")
    print("=" * 60)
    
    return all_passed

def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("RL VIDEO COMPRESSION - INSTALLATION VERIFICATION")
    print("=" * 60)
    print("This script verifies your environment is ready for reproduction\n")
    
    results = {}
    
    # Required checks
    results['python'] = check_python_version()
    results['pytorch'] = check_pytorch()
    results['packages'] = check_required_packages()
    results['directories'] = check_directory_structure()
    
    # Optional checks
    check_model_files()  # Informational only
    results['dataset'] = check_dataset()  # Informational only
    
    # Print summary
    success = print_summary(results)
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
