"""Download GPT-2 (or other small models) from multiple fallback sources.

When HuggingFace returns 403 (e.g. corporate proxy / Zscaler), this script
tries alternative sources in order until one succeeds.

Usage examples::

    # Download GPT-2 124M with default settings
    python scripts/download_model.py

    # Download to a custom directory
    python scripts/download_model.py --output-dir data/models/gpt2-medium --model-id openai-community/gpt2-medium

    # Force re-download even if directory exists
    python scripts/download_model.py --force
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _inject_truststore() -> None:
    """Use Windows certificate store for SSL if truststore is available."""
    try:
        import truststore
        truststore.inject_into_ssl()
        log.info("Using Windows certificate store via truststore")
    except ImportError:
        log.debug("truststore not installed, using default SSL")
    except Exception as exc:
        log.debug("truststore injection failed: %s", exc)


def _patch_hf_endpoint(endpoint: str | None) -> None:
    """Patch huggingface_hub endpoint at runtime (survives import caching)."""
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    try:
        import huggingface_hub.constants as hf_const
        if endpoint:
            hf_const.ENDPOINT = endpoint
        else:
            hf_const.ENDPOINT = "https://huggingface.co"
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Source 1: ModelScope (most reliable behind corporate proxies)
# ---------------------------------------------------------------------------

MODELSCOPE_MAP: dict[str, str] = {
    "openai-community/gpt2": "AI-ModelScope/gpt2",
    "gpt2": "AI-ModelScope/gpt2",
    "openai-community/gpt2-medium": "AI-ModelScope/gpt2-medium",
    "gpt2-medium": "AI-ModelScope/gpt2-medium",
}

# Only download files needed for PyTorch inference
MODELSCOPE_ALLOW = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def try_modelscope(model_id: str, output_dir: Path) -> bool:
    """Download from Alibaba ModelScope (filtered to PyTorch files only)."""
    log.info("[1/3] Trying ModelScope ...")
    try:
        from modelscope import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        log.warning(
            "  -> modelscope not installed. To enable this source:\n"
            "     uv pip install modelscope"
        )
        return False

    ms_id = MODELSCOPE_MAP.get(model_id)
    if ms_id is None:
        log.warning("  -> No ModelScope mapping for '%s'. Skipping.", model_id)
        return False

    try:
        cache_dir = snapshot_download(
            ms_id,
            allow_patterns=MODELSCOPE_ALLOW,
        )
        log.info("  -> Downloaded to ModelScope cache: %s", cache_dir)

        # Copy files directly from cache to output (avoids loading model into memory)
        cache_path = Path(cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in cache_path.iterdir():
            if f.is_file() and not f.name.startswith("."):
                shutil.copy2(f, output_dir / f.name)
                copied.append(f.name)
        log.info("  -> Copied %d files: %s", len(copied), copied)

        # Clean up cache to save disk space
        shutil.rmtree(cache_path, ignore_errors=True)
        log.info("  -> Cleaned ModelScope cache")

        log.info("  -> Success via ModelScope")
        return True
    except Exception as exc:
        log.warning("  -> ModelScope failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Source 2: HuggingFace Mirror
# ---------------------------------------------------------------------------

def try_hf_mirror(model_id: str, output_dir: Path) -> bool:
    """Download via hf-mirror.com (Chinese HF mirror)."""
    log.info("[2/3] Trying HuggingFace Mirror (hf-mirror.com) ...")
    original_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        _patch_hf_endpoint("https://hf-mirror.com")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        log.info("  -> Success via HF Mirror")
        return True
    except Exception as exc:
        log.warning("  -> HF Mirror failed: %s", exc)
        return False
    finally:
        _patch_hf_endpoint(original_endpoint)


# ---------------------------------------------------------------------------
# Source 3: HuggingFace Direct
# ---------------------------------------------------------------------------

def try_hf_direct(model_id: str, output_dir: Path) -> bool:
    """Download from standard HuggingFace (in case 403 is intermittent)."""
    log.info("[3/3] Trying HuggingFace directly ...")
    original_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        _patch_hf_endpoint(None)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        log.info("  -> Success via HuggingFace direct")
        return True
    except Exception as exc:
        log.warning("  -> HuggingFace direct failed: %s", exc)
        return False
    finally:
        _patch_hf_endpoint(original_endpoint)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model(output_dir: Path) -> bool:
    """Load model + tokenizer from disk and print basic info."""
    log.info("Verifying downloaded model at %s ...", output_dir)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))

        n_params = sum(p.numel() for p in model.parameters())
        log.info("  Model type:    %s", type(model).__name__)
        log.info("  Parameters:    %s (%.1fM)", f"{n_params:,}", n_params / 1e6)
        log.info("  Hidden size:   %s", model.config.hidden_size)
        log.info("  Vocab size:    %s", model.config.vocab_size)
        log.info("  Tokenizer:     %s (%d tokens)", type(tokenizer).__name__, len(tokenizer))

        tokens = tokenizer.encode("Hello world")
        log.info("  Tokenize test: 'Hello world' -> %s", tokens)

        return True
    except Exception as exc:
        log.error("  Verification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GPT-2 (or similar) from multiple fallback sources."
    )
    parser.add_argument(
        "--model-id",
        default="openai-community/gpt2",
        help="HuggingFace model ID (default: openai-community/gpt2)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/gpt2",
        help="Local directory to save model (default: data/models/gpt2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output directory already exists",
    )
    args = parser.parse_args()

    # Inject truststore early so all HTTPS requests use Windows cert store
    _inject_truststore()

    # Resolve relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    # Check for existing download
    if output_dir.exists() and (output_dir / "config.json").exists():
        if not args.force:
            log.info("Model already exists at %s (use --force to re-download)", output_dir)
            verify_model(output_dir)
            return
        else:
            log.info("Removing existing model at %s (--force)", output_dir)
            shutil.rmtree(output_dir)

    # Try each source in order (ModelScope first — most reliable behind proxies)
    sources = [
        try_modelscope,
        try_hf_mirror,
        try_hf_direct,
    ]

    success = False
    for source_fn in sources:
        try:
            if source_fn(args.model_id, output_dir):
                success = True
                break
        except Exception as exc:
            log.error("Unexpected error in %s: %s", source_fn.__name__, exc)
            continue

    if not success:
        log.error("=" * 60)
        log.error("All download sources failed.")
        log.error("")
        log.error("MANUAL FALLBACK: Download on another machine and copy here:")
        log.error("  1. On an unrestricted machine, run:")
        log.error("     python -c \"")
        log.error("       from transformers import AutoModelForCausalLM, AutoTokenizer")
        log.error("       m = AutoModelForCausalLM.from_pretrained('%s')", args.model_id)
        log.error("       t = AutoTokenizer.from_pretrained('%s')", args.model_id)
        log.error("       m.save_pretrained('gpt2_export')")
        log.error("       t.save_pretrained('gpt2_export')")
        log.error("     \"")
        log.error("  2. Copy the 'gpt2_export' folder to: %s", output_dir)
        log.error("=" * 60)
        sys.exit(1)

    # Verify
    if not verify_model(output_dir):
        log.error("Download succeeded but verification failed. Check the files at %s", output_dir)
        sys.exit(1)

    log.info("")
    log.info("Done! Model saved to: %s", output_dir)
    log.info("Use in config:  llm.name: %s", output_dir.relative_to(repo_root))


if __name__ == "__main__":
    main()
