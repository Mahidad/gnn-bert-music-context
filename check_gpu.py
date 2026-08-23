"""
Run this FIRST, before train_task1.py.

It verifies that PyTorch can actually see your GPU. If this prints
"CPU-only build" you installed the wrong torch package -- see the README
for the fix. Catching that here takes 5 seconds; discovering it later
means realising mid-training that your RTX 4060 was never being used.
"""

import torch


def main():
    print("=" * 55)
    print("PyTorch / GPU environment check")
    print("=" * 55)
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Built with CUDA:      {torch.version.cuda}")
    print(f"CUDA available:       {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detected:         {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM:           {total_vram:.1f} GB")

        # Actually run something on the GPU -- "is_available() == True" can
        # still be followed by a driver error on the first real operation,
        # so this proves the whole path works end to end.
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"Test matmul on GPU:   OK (result shape {tuple(y.shape)})")
        print("\nYou're good to go. Run: python train_task1.py")
    else:
        print("\n--- PROBLEM ---")
        if torch.version.cuda is None:
            print("You have the CPU-only build of PyTorch installed.")
            print("Fix: uninstall it and reinstall with the CUDA index URL:")
            print("  pip uninstall -y torch")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("PyTorch has CUDA support compiled in, but no GPU is visible.")
            print("This usually means your NVIDIA driver is missing or outdated.")
            print("Fix: update your GeForce driver from nvidia.com, then reboot.")
            print("Check the driver is alive by running:  nvidia-smi")


if __name__ == "__main__":
    main()
