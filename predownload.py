#!/usr/bin/env python3
"""
Model Pre-download Script for vLLM

Downloads models from HuggingFace without vLLM's memory checks.
Useful for downloading large models that vLLM thinks won't fit.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, List
import time

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    import fnmatch
except ImportError:
    print("Error: huggingface_hub not installed. Install with:")
    print("pip install huggingface_hub")
    sys.exit(1)

def get_profile_models(profiles_dir: str = "profiles") -> List[tuple]:
    """Get all models from profile files."""
    profiles_path = Path(profiles_dir)
    models = []

    if not profiles_path.exists():
        print(f"Warning: Profiles directory {profiles_dir} not found")
        return models

    for profile_file in profiles_path.glob("*.yaml"):
        try:
            with open(profile_file, 'r') as f:
                config = yaml.safe_load(f)
                model_name = config.get('model', '')
                profile_name = config.get('name', profile_file.stem)
                if model_name:
                    models.append((profile_name, model_name))
        except Exception as e:
            print(f"Warning: Could not read {profile_file}: {e}")

    return models

def get_optimal_download_patterns(model_id: str, token: Optional[str] = None) -> List[str]:
    """Get optimal download patterns using vLLM's logic to avoid duplicates."""
    try:
        # Get file list from HuggingFace repo
        files = list_repo_files(repo_id=model_id, token=token)

        # vLLM's pattern priority order (safetensors first)
        weight_patterns = ["*.safetensors", "*.bin"]
        config_patterns = ["*.json", "*.txt", "*.py"]

        # Use vLLM's logic: first pattern that matches wins
        selected_weight_pattern = None
        for pattern in weight_patterns:
            matching = fnmatch.filter(files, pattern)
            if len(matching) > 0:
                selected_weight_pattern = pattern
                format_name = "safetensors" if pattern == "*.safetensors" else "PyTorch bin"
                print(f"🎯 Using {format_name} format ({len(matching)} files)")
                break

        # Combine selected weight pattern with config patterns
        if selected_weight_pattern:
            return [selected_weight_pattern] + config_patterns
        else:
            print("⚠️ No standard weight files found, including all patterns")
            return weight_patterns + config_patterns

    except Exception as e:
        print(f"⚠️ Could not detect optimal patterns for {model_id}: {e}")
        # Fallback to safetensors only (vLLM default)
        return ["*.safetensors", "*.json", "*.txt", "*.py"]

def download_model(model_id: str, force: bool = False, token: Optional[str] = None) -> bool:
    """Download a model from HuggingFace."""
    print(f"\n🔄 Downloading {model_id}...")

    try:
        # Check if already downloaded (unless forcing)
        if not force:
            status, size = check_download_status(model_id)
            if status == "✅ Downloaded":
                print(f"✅ {model_id} already downloaded ({size})")
                return True
            elif status in ["⚠️ Incomplete", "⚠️ Partial"]:
                print(f"⚠️ {model_id} partially downloaded ({size}), completing download...")

        if force:
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{model_id.replace("/", "--")}'
            if cache_dir.exists():
                print(f"🔄 Force re-downloading {model_id}...")

        # Get optimal download patterns using vLLM's logic
        allow_patterns = get_optimal_download_patterns(model_id, token)

        # Download with progress
        start_time = time.time()

        downloaded_path = snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.md", "*.git*", "*.DS_Store"],
        )

        elapsed = time.time() - start_time
        print(f"✅ {model_id} downloaded successfully in {elapsed:.1f}s")
        print(f"📂 Location: {downloaded_path}")

        return True

    except RepositoryNotFoundError:
        print(f"❌ Model {model_id} not found on HuggingFace")
        return False
    except GatedRepoError:
        print(f"🔒 Model {model_id} is gated. You need to:")
        print(f"   1. Accept the license at https://huggingface.co/{model_id}")
        print(f"   2. Login with: huggingface-cli login")
        return False
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return False

def check_download_status(model_id: str) -> tuple:
    """Check if model is fully downloaded by looking for main model files."""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{model_id.replace("/", "--")}'

    if not cache_dir.exists():
        return "📥 Not downloaded", "0GB"

    # Look for model files in snapshots
    model_files = []
    for pattern in ['*.safetensors', '*.bin', '*.pt', '*.pth']:
        model_files.extend(list(cache_dir.glob(f'snapshots/*/{pattern}')))

    if not model_files:
        return "⚠️ Incomplete", "0GB"

    # Calculate total size of model files
    total_size = 0
    for file_path in model_files:
        if file_path.is_file():
            try:
                # Resolve symlinks and get actual file size
                real_path = file_path.resolve()
                size = real_path.stat().st_size
                if size > 100 * 1024 * 1024:  # Only count files > 100MB
                    total_size += size
            except:
                continue

    size_gb = total_size / (1024**3)

    if size_gb < 1:
        return "⚠️ Incomplete", f"{size_gb*1024:.0f}MB"
    elif size_gb < 5:  # Likely incomplete for large models
        return "⚠️ Partial", f"{size_gb:.1f}GB"
    else:
        return "✅ Downloaded", f"{size_gb:.1f}GB"

def show_profile_models():
    """Show all models from profiles."""
    models = get_profile_models()

    if not models:
        print("No models found in profiles directory")
        return

    print("\n📋 Available models from profiles:")
    print("=" * 75)
    print(f"{'#':<3} {'Model ID':<50} {'Size':<8} {'Status'}")
    print("-" * 75)

    for i, (profile_name, model_id) in enumerate(models, 1):
        status, size = check_download_status(model_id)
        print(f"{i:2d}. {model_id:<50} {size:<8} {status}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(
        description="Pre-download models for vLLM without memory checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predownload.py --list                           # Show all profile models
  python predownload.py --all                            # Download all profile models
  python predownload.py --model mistralai/Magistral-Small-2509
  python predownload.py --profile qwen3-30b-a3b-gptq-int4
  python predownload.py --model gpt2 --force             # Force re-download
        """
    )

    parser.add_argument("--model", help="Specific model to download (e.g., mistralai/Magistral-Small-2509)")
    parser.add_argument("--profile", help="Download model from specific profile")
    parser.add_argument("--all", action="store_true", help="Download all models from profiles")
    parser.add_argument("--list", action="store_true", help="List all models from profiles")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--token", help="HuggingFace token for gated models")
    parser.add_argument("--profiles-dir", default="profiles", help="Directory containing profile files")

    args = parser.parse_args()

    # Show usage if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        show_profile_models()
        return

    # List models
    if args.list:
        show_profile_models()
        return

    # Get HF token from environment if not provided
    token = args.token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')

    success_count = 0
    total_count = 0

    if args.model:
        # Download specific model
        total_count = 1
        if download_model(args.model, args.force, token):
            success_count = 1

    elif args.profile:
        # Download from specific profile
        profiles_path = Path(args.profiles_dir)
        profile_file = profiles_path / f"{args.profile}.yaml"

        if not profile_file.exists():
            print(f"❌ Profile {args.profile} not found at {profile_file}")
            return

        try:
            with open(profile_file, 'r') as f:
                config = yaml.safe_load(f)
                model_id = config.get('model', '')

                if not model_id:
                    print(f"❌ No model specified in profile {args.profile}")
                    return

                total_count = 1
                if download_model(model_id, args.force, token):
                    success_count = 1

        except Exception as e:
            print(f"❌ Error reading profile {args.profile}: {e}")
            return

    elif args.all:
        # Download all models from profiles
        models = get_profile_models(args.profiles_dir)

        if not models:
            print("❌ No models found in profiles")
            return

        print(f"📥 Downloading {len(models)} models from profiles...")

        for profile_name, model_id in models:
            total_count += 1
            print(f"\n[{total_count}/{len(models)}] Profile: {profile_name}")
            if download_model(model_id, args.force, token):
                success_count += 1

    else:
        parser.print_help()
        return

    # Summary
    if total_count > 0:
        print(f"\n📊 Summary: {success_count}/{total_count} models downloaded successfully")

        if success_count < total_count:
            print("💡 For gated models, make sure to:")
            print("   1. Accept licenses on HuggingFace")
            print("   2. Login with: huggingface-cli login")

if __name__ == "__main__":
    main()