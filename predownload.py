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
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
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

def download_model(model_id: str, force: bool = False, token: Optional[str] = None) -> bool:
    """Download a model from HuggingFace."""
    print(f"\n🔄 Downloading {model_id}...")

    try:
        # Check if already downloaded
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{model_id.replace("/", "--")}'

        if cache_dir.exists() and not force:
            print(f"✅ {model_id} already cached at {cache_dir}")
            return True

        if force and cache_dir.exists():
            print(f"🔄 Force re-downloading {model_id}...")

        # Download with progress
        start_time = time.time()

        downloaded_path = snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.py"],  # Skip unnecessary files
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

def show_profile_models():
    """Show all models from profiles."""
    models = get_profile_models()

    if not models:
        print("No models found in profiles directory")
        return

    print("\n📋 Available models from profiles:")
    print("=" * 80)
    for i, (profile_name, model_id) in enumerate(models, 1):
        # Check if already downloaded
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{model_id.replace("/", "--")}'
        status = "✅ Downloaded" if cache_dir.exists() else "📥 Not downloaded"

        print(f"{i:2d}. {profile_name:<35} {model_id:<45} {status}")
    print("=" * 80)

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