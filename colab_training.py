#!/usr/bin/env python3
"""
Google Colab Installation and End-to-End Training Script for SensorLLM

This script automates the complete setup and training pipeline for SensorLLM in Google Colab:
1. Detects Colab environment and installs dependencies
2. Clones the SensorLLM repository (if needed)
3. Installs the package
4. Generates synthetic sensor data
5. Runs an end-to-end training experiment
6. (Optional) Evaluates the trained model

Usage in Google Colab:
    !pip install -q gdown
    !gdown https://drive.google.com/uc?id=FILE_ID -O colab_training.py
    !python colab_training.py

Or run directly:
    !git clone https://github.com/your-org/sensorllm.git /content/sensorllm
    %cd /content/sensorllm
    !python colab_training.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def log_header(msg: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def log_info(msg: str) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}➜ {msg}{Colors.END}")


def log_success(msg: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def log_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def log_error(msg: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def run_cmd(cmd: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """Execute a shell command."""
    if not quiet:
        log_info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=quiet)
    if check and result.returncode != 0:
        log_error(f"Command failed with exit code {result.returncode}")
        if quiet and result.stderr:
            print(result.stderr.decode())
        sys.exit(1)
    return result


def is_colab() -> bool:
    """Check if we're running in Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            log_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log_warning("CUDA not available. Training will be slow on CPU.")
        return cuda_available
    except ImportError:
        return False


def setup_environment(repo_root: Path, data_root: Path, output_root: Path) -> None:
    """Set up environment variables."""
    log_header("Setting Up Environment")

    # Create directories if they don't exist
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["SENSORLLM_DATA_ROOT"] = str(data_root)
    os.environ["SENSORLLM_OUTPUT_ROOT"] = str(output_root)
    os.environ["WANDB_MODE"] = "offline"  # Disable W&B for Colab by default

    log_success(f"Data root: {data_root}")
    log_success(f"Output root: {output_root}")
    log_info("Weights & Biases set to offline mode (set WANDB_MODE=online to enable)")

    # Create .env file
    env_file = repo_root / ".env"
    if not env_file.exists():
        env_content = f"""# Auto-generated environment file for Colab training
SENSORLLM_DATA_ROOT={data_root}
SENSORLLM_OUTPUT_ROOT={output_root}
WANDB_PROJECT=sensorllm
WANDB_MODE=offline
"""
        env_file.write_text(env_content)
        log_success(f"Created .env file: {env_file}")


def install_package(repo_root: Path) -> None:
    """Install the SensorLLM package."""
    log_header("Installing SensorLLM Package")

    # Change to repo root
    original_cwd = os.getcwd()
    os.chdir(repo_root)

    try:
        # Install in editable mode with dev dependencies
        run_cmd(
            "pip install -q -e '.[dev]'",
            quiet=True,
        )
        log_success("SensorLLM package installed successfully")
    finally:
        os.chdir(original_cwd)


def generate_synthetic_data(repo_root: Path, data_root: Path, num_samples: int = 20) -> None:
    """Generate synthetic sensor data for training."""
    log_header("Generating Synthetic Sensor Data")

    original_cwd = os.getcwd()
    os.chdir(repo_root)

    try:
        cmd = (
            f"python scripts/generate_synthetic_data.py "
            f"--data-root {data_root} "
            f"--samples {num_samples} "
            f"--summary"
        )
        run_cmd(cmd)
        log_success("Synthetic data generated successfully")
    finally:
        os.chdir(original_cwd)


def train_model(
    repo_root: Path,
    config_path: str = "configs/experiments/exp001_cnn1d_linear.yaml",
    max_steps: int = 1000,
    batch_size: int = 8,
) -> None:
    """Run the training experiment."""
    log_header("Running Training Experiment")

    original_cwd = os.getcwd()
    os.chdir(repo_root)

    try:
        # Build training command with overrides for Colab constraints
        cmd = f"python scripts/train.py --config {config_path} --override "
        overrides = [
            f"training.max_steps={max_steps}",
            f"data.batch_size={batch_size}",
            "training.eval_steps=100",
            "training.save_steps=200",
            "training.logging_steps=20",
        ]
        cmd += " ".join(overrides)

        run_cmd(cmd)
        log_success("Training completed successfully")
    finally:
        os.chdir(original_cwd)


def verify_installation() -> bool:
    """Verify that all required packages are installed."""
    log_header("Verifying Installation")

    required_packages = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "h5py",
        "yaml",
        "einops",
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            log_success(f"✓ {package}")
        except ImportError:
            log_error(f"✗ {package} not found")
            all_installed = False

    return all_installed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SensorLLM Google Colab Setup and Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python colab_training.py

  # Custom number of training steps and batch size
  python colab_training.py --max-steps 2000 --batch-size 16

  # Use a different experiment config
  python colab_training.py --config configs/experiments/exp002_transformer_qformer.yaml

  # Generate more synthetic samples
  python colab_training.py --num-samples 50
        """,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to SensorLLM repository root (default: current directory)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to data directory (default: {repo-root}/data)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Path to output directory (default: {repo-root}/outputs)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp004_cnn1d_linear_gpt2.yaml",
        help="Experiment config to use (default: exp004_cnn1d_linear_gpt2.yaml — uses tiny local GPT-2, no auth needed)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000). Use smaller values on limited resources.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8). Reduce if out of memory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of synthetic samples per class (default: 20)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only generate synthetic data",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto). Use 'cpu' if CUDA issues occur.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    repo_root = args.repo_root.resolve()
    data_root = args.data_root or (repo_root / "data")
    output_root = args.output_root or (repo_root / "outputs")

    log_header("SensorLLM Google Colab Installation & Training")

    # Check environment
    in_colab = is_colab()
    if in_colab:
        log_success("Running in Google Colab")
    else:
        log_warning("Not running in Google Colab (local environment)")

    # Check CUDA
    cuda_available = check_cuda()

    # Update device_map based on CUDA availability
    if not cuda_available and args.device_map == "auto":
        log_warning("Forcing device_map='cpu' since CUDA is not available")
        args.device_map = "cpu"

    # Set up environment
    setup_environment(repo_root, data_root, output_root)

    # Install package
    install_package(repo_root)

    # Verify installation
    if not verify_installation():
        log_error("Some required packages are missing")
        return 1

    # Generate synthetic data
    try:
        generate_synthetic_data(repo_root, data_root, args.num_samples)
    except Exception as e:
        log_error(f"Failed to generate synthetic data: {e}")
        return 1

    # Train model (unless skipped)
    if not args.skip_train:
        try:
            train_model(
                repo_root,
                config_path=args.config,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
            )
        except Exception as e:
            log_error(f"Training failed: {e}")
            return 1

        # Print summary
        log_header("Training Complete!")
        log_success(f"Outputs saved to: {output_root}")
        log_info(
            f"Check {output_root}/{args.config.split('/')[-1].replace('.yaml', '')} "
            "for experiment results"
        )
    else:
        log_header("Synthetic Data Generated")
        log_info("Run 'python scripts/train.py' to start training")

    return 0


if __name__ == "__main__":
    sys.exit(main())
