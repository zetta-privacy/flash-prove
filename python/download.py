#!/usr/bin/env python3
"""Download OLMoE-1B-7B (config + weights) from HuggingFace."""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HF model's config + weights.")
    parser.add_argument("--repo", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--out", default="./olmoe")
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "tokenizer*.json",
            "special_tokens_map.json",
        ],
    )
    print(f"downloaded {args.repo} -> {args.out}")


if __name__ == "__main__":
    main()
