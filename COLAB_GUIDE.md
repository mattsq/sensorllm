# SensorLLM Google Colab Guide

This guide explains how to install and train SensorLLM in Google Colab using synthetic sensor data.

## Quick Start

### Option 1: Interactive Notebook (Recommended)

1. **Open in Colab**: Click the button below or go to [Google Colab](https://colab.research.google.com) and upload the notebook

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/sensorllm/blob/main/SensorLLM_Colab_Training.ipynb)

2. **Select GPU Runtime** (important!):
   - Go to **Runtime → Change runtime type**
   - Select **GPU** (T4 is recommended)
   - Click **Save**

3. **Run all cells** in order from top to bottom
   - First run will take ~10-15 minutes to download models and dependencies
   - Training will take ~15-25 minutes
   - You'll see a summary of results at the end

### Option 2: Command-Line Script

1. Clone the repository in Colab:
   ```bash
   !git clone https://github.com/your-org/sensorllm.git /content/sensorllm
   %cd /content/sensorllm
   ```

2. Run the automated setup and training script:
   ```bash
   !python colab_training.py --max-steps 500 --batch-size 8
   ```

3. View results:
   ```bash
   !ls -la /content/sensorllm_outputs/runs/
   ```

## What Gets Installed

The installation includes:

- **PyTorch & CUDA** - Deep learning framework
- **Transformers** - HuggingFace model library
- **SensorLLM** - The main package
- **Dependencies**: scipy, librosa, h5py, accelerate, peft, wandb, etc.

**Total size**: ~5 GB (mostly LLM models)
**Installation time**: 5-10 minutes (first time only)

## What The Pipeline Does

```
1. Install dependencies
      ↓
2. Generate synthetic sensor data
   - Vibration signals (normal & faulty)
   - Temperature data
   - IMU readings
   - Pressure measurements
      ↓
3. Train end-to-end model
   - Input: Raw sensor time-series (B, C, L)
   - Encoder: CNN1D (dilated convolution)
   - Adapter: Linear projection
   - LLM: Tiny GPT-2 (local, no auth needed)
   - Output: Learned alignment between sensors → LLM
      ↓
4. Save results
   - Trained checkpoints
   - Metrics (loss, learning rate curves)
   - Configuration (frozen)
```

## File Descriptions

### `colab_training.py`
Standalone Python script for automated setup and training. Useful for:
- CI/CD pipelines
- Batch training multiple experiments
- Command-line usage

**Usage:**
```bash
python colab_training.py --max-steps 1000 --batch-size 8
python colab_training.py --config configs/experiments/exp002_transformer_qformer.yaml
python colab_training.py --num-samples 50 --skip-train  # Only generate data
```

**Key arguments:**
- `--max-steps`: Maximum training steps (default: 1000)
- `--batch-size`: Batch size (default: 8, reduce if OOM)
- `--num-samples`: Synthetic samples per class (default: 20)
- `--config`: Experiment config (default: exp001_cnn1d_linear.yaml)
- `--skip-train`: Only generate data, don't train

### `SensorLLM_Colab_Training.ipynb`
Interactive Jupyter notebook with:
- Step-by-step installation
- Detailed explanations
- Training monitoring
- Results visualization
- Metrics plotting
- Troubleshooting guide

## Hyperparameter Tuning

### Reduce Memory Usage

For small GPUs (T4 with 16 GB):

```bash
python colab_training.py \
  --batch-size 4 \
  --max-steps 1000
```

For CPU-only (very slow):

```python
os.environ["DEVICE_MAP"] = "cpu"
```

### Increase Training Steps

For more thorough training:

```bash
python colab_training.py --max-steps 5000
```

### Larger Dataset

Generate more synthetic samples:

```bash
python colab_training.py --num-samples 100
```

## Monitoring Training

### View Real-Time Logs

In the notebook, logs print automatically. For the script:

```bash
tail -f /content/sensorllm_outputs/runs/exp001_cnn1d_linear/*/logs/*.log
```

### Plot Metrics

After training completes, the notebook has a cell to plot:
- Training loss over time
- Validation loss
- Learning rate schedule

### Access Checkpoints

```python
from pathlib import Path

output_root = Path("/content/sensorllm_outputs")
run_dir = list(output_root.glob("runs/*/*/"))[-1]  # Latest run

# List checkpoints
checkpoints = sorted(run_dir.glob("checkpoint-*"))
best_model = run_dir / "best_model"

print(f"Best model: {best_model}")
print(f"Checkpoints: {[c.name for c in checkpoints]}")
```

## Different Experiments

Try other encoder/adapter combinations:

### Experiment 2: Transformer + Q-Former

```bash
python colab_training.py \
  --config configs/experiments/exp002_transformer_qformer.yaml \
  --max-steps 1000
```

### Experiment 3: PatchTST + Perceiver

```bash
python colab_training.py \
  --config configs/experiments/exp003_patchtst_perceiver.yaml \
  --max-steps 1000
```

See `configs/experiments/` for all available experiment configurations.

## Output Structure

After training, outputs are organized as:

```
/content/sensorllm_outputs/
└── runs/
    └── exp001_cnn1d_linear/
        └── 2024-01-15_1200/
            ├── config.yaml              # Frozen config (reproducibility)
            ├── metrics.jsonl            # Training metrics (one per line)
            ├── logs/                    # Detailed logs
            ├── checkpoint-100/          # Intermediate checkpoints
            ├── checkpoint-200/
            └── best_model/              # Symlink to best checkpoint
                ├── adapter.safetensors
                ├── encoder.safetensors
                ├── config.json
                └── ...
```

## Downloading Results

### Download Model and Metrics

```python
import shutil
shutil.make_archive(
    "/content/sensorllm_results",
    "zip",
    "/content/sensorllm_outputs"
)
files.download("/content/sensorllm_results.zip")
```

### Download Only Best Model

```python
import shutil
best_model_path = "/content/sensorllm_outputs/runs/exp001_cnn1d_linear/.../best_model"
shutil.make_archive(
    "/content/best_model",
    "zip",
    best_model_path.parent,
    best_model_path.name
)
files.download("/content/best_model.zip")
```

## Running Evaluation

After training, evaluate on test set:

```bash
cd /content/sensorllm
python scripts/evaluate.py \
  --config configs/experiments/exp001_cnn1d_linear.yaml \
  --checkpoint /content/sensorllm_outputs/runs/exp001_cnn1d_linear/*/best_model
```

## Running Inference

Make predictions on sensor data:

```bash
cd /content/sensorllm
python scripts/infer.py \
  --checkpoint /content/sensorllm_outputs/runs/exp001_cnn1d_linear/*/best_model \
  --sensor-file /content/sensorllm_data/raw/synthetic/vibration_normal_0000.h5 \
  --prompt "Describe the sensor state and any anomalies."
```

## Troubleshooting

### Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```bash
   python colab_training.py --batch-size 4
   ```

2. Reduce window size:
   ```bash
   python colab_training.py --max-steps 500 --batch-size 8
   ```

3. Use CPU (slow):
   ```python
   os.environ["DEVICE_MAP"] = "cpu"
   ```

### Model Download Hangs

**Issue**: Model downloads are slow or timeout

**Solutions**:
1. Use a smaller LLM in the config
2. Increase the timeout in your internet settings
3. Try a different Colab session

### No GPU Available

**Error**: "CUDA not available"

**Solution**:
1. Go to **Runtime → Change runtime type**
2. Select **GPU** (T4 recommended)
3. Click **Save** and run again

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'sensorllm'`

**Solution**:
```bash
cd /content/sensorllm
pip install -e .
```

### Synthetic Data Not Generated

**Error**: "No training samples found"

**Solution**:
```bash
python scripts/generate_synthetic_data.py \
  --data-root /content/sensorllm_data \
  --samples 20 \
  --summary
```

## Advanced Usage

### Custom Dataset

To use your own sensor data:

1. Format as HDF5 files (see `docs/data_format.md`)
2. Place in `data/raw/your_dataset/`
3. Create split files in `data/splits/` (JSONL format)
4. Update config to point to your data

### Stage 2 Fine-Tuning

After Stage 1 (alignment) training, run Stage 2 with LoRA:

```bash
# Edit configs/experiments/exp001_cnn1d_linear.yaml:
# - training.stage: 2
# - model.llm.lora.enabled: true
# - training.max_steps: 5000

python scripts/train.py --config configs/experiments/exp001_stage2_finetuning.yaml
```

### Multi-GPU Training

Automatic with Accelerate:

```bash
accelerate launch scripts/train.py --config configs/experiments/exp001_cnn1d_linear.yaml
```

## Performance Notes

**Typical Colab Performance** (T4 GPU, batch_size=8):

| Encoder | Adapter | Steps | Time | Memory |
|---------|---------|-------|------|--------|
| CNN1D | Linear | 500 | 15 min | 10 GB |
| CNN1D | Linear | 1000 | 25 min | 10 GB |
| Transformer | Q-Former | 500 | 20 min | 12 GB |
| PatchTST | Perceiver | 500 | 25 min | 14 GB |

**Tips for faster training**:
- Use smaller batch sizes (4-8)
- Reduce window size (2048 instead of 4096)
- Reduce max_steps for quick iteration
- Use offline W&B mode (default)

## Citation

If you use SensorLLM in your research, please cite:

```bibtex
@article{sensorllm2024,
  title={SensorLLM: Fusing Aircraft Sensor Data with Large Language Models},
  author={Your Name and Collaborators},
  journal={Your Journal},
  year={2024}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full documentation in `docs/`
3. Open an issue on GitHub: https://github.com/your-org/sensorllm/issues

## License

SensorLLM is released under the MIT License. See `LICENSE` for details.
