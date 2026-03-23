"""
Setup Verification Script
Checks if all dependencies are correctly installed
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info

    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (not installed)")
        return False


def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    print("\nChecking PyTorch...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (device: {torch.cuda.get_device_name(0)})")
            print(f"    - CUDA version: {torch.version.cuda}")
            print(f"    - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")

        return True
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")

    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        'models/ris_net.py',
        'src/server.py',
        'src/client.py',
        'src/dataset_utils.py',
        'utils/metrics.py',
        'utils/plotting.py'
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            all_exist = False

    return all_exist


def check_directories():
    """Check/create required directories"""
    print("\nChecking directories...")

    required_dirs = [
        'data',
        'results',
        'models/saved',
        'plots',
        'metrics'
    ]

    for dir_name in required_dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
        else:
            print(f"  ✓ {dir_name}/ exists")

    return True


def run_minimal_test():
    """Run a minimal functionality test"""
    print("\nRunning minimal functionality test...")

    try:
        import torch
        import numpy as np
        from models.ris_net import RISNet

        # Create small model
        model = RISNet(input_dim=50, num_elements=16, hidden_dim=64, num_layers=2)

        # Test forward pass
        x = torch.randn(4, 50)
        output = model(x)

        assert output.shape == (4, 16), "Output shape mismatch"
        assert output.min() >= 0 and output.max() <= 2 * np.pi, "Output range incorrect"

        print(f"  ✓ Model creation and forward pass")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")

        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def check_gpu_memory():
    """Check available GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\nGPU Memory Status:")
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                free = total_mem - allocated
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}):")
                print(f"    Total: {total_mem:.2f} GB")
                print(f"    Free: {free:.2f} GB")
                print(f"    Allocated: {allocated:.2f} GB")
    except:
        pass


def estimate_training_time():
    """Estimate training time"""
    print("\nEstimated Training Time:")

    try:
        import torch

        configs = [
            ("Small (4 tiles, 50 rounds)", 4, 50),
            ("Medium (8 tiles, 100 rounds)", 8, 100),
            ("Large (16 tiles, 200 rounds)", 16, 200)
        ]

        device_str = "GPU" if torch.cuda.is_available() else "CPU"

        for name, tiles, rounds in configs:
            # Rough estimate: ~0.1s per round per tile on GPU, ~0.5s on CPU
            time_per_round = 0.1 if torch.cuda.is_available() else 0.5
            total_time = tiles * rounds * time_per_round / 60  # Convert to minutes

            print(f"  {name}: ~{total_time:.1f} minutes ({device_str})")

    except:
        print("  Unable to estimate (PyTorch not installed)")


def print_recommendations():
    """Print optimization recommendations"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("\n⚠️  No GPU detected. Training will be slower.")
            print("   Recommendations:")
            print("   - Reduce NUM_TILES in config.py")
            print("   - Reduce FL_ROUNDS in config.py")
            print("   - Reduce TRAIN_SAMPLES in config.py")
            print("   - Consider using Google Colab or cloud GPU")
        else:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < 4:
                print("\n⚠️  Limited GPU memory detected.")
                print("   Recommendations:")
                print("   - Reduce BATCH_SIZE in config.py")
                print("   - Reduce HIDDEN_DIM in config.py")
            else:
                print("\n✓ GPU configuration looks good!")
                print("  You can use default settings for best results.")

    except:
        pass


def main():
    """Main setup check"""
    print("=" * 60)
    print("FEDERATED RIS SETUP VERIFICATION")
    print("=" * 60)

    checks = []

    # Python version
    checks.append(check_python_version())

    # Core packages
    print("\nChecking core packages...")
    checks.append(check_package('torch'))
    checks.append(check_package('numpy'))
    checks.append(check_package('scipy'))
    checks.append(check_package('matplotlib'))
    checks.append(check_package('seaborn'))
    checks.append(check_package('pandas'))

    # PyTorch details
    checks.append(check_pytorch())

    # Project structure
    checks.append(check_project_structure())

    # Directories
    checks.append(check_directories())

    # Minimal test
    checks.append(run_minimal_test())

    # GPU info
    check_gpu_memory()

    # Time estimates
    estimate_training_time()

    # Recommendations
    print_recommendations()

    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)

    passed = sum(checks)
    total = len(checks)

    print(f"\nChecks passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYou're ready to start training:")
        print("  python main.py")
        print("\nOr run component tests:")
        print("  python test_components.py")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Install missing packages:")
        print("  pip install -r requirements.txt")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
