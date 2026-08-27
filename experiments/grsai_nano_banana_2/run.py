from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_toolkit import grsai

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call only GRS.AI nano-banana-2 and save one generated image."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--size", choices=("1K", "2K", "4K"), default="1K")
    parser.add_argument(
        "--ratio",
        choices=tuple(sorted(grsai.images.SUPPORTED_ASPECT_RATIOS)),
        default="1:1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "outputs" / "result.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = grsai.images.generate(
        model="grsai-image",
        prompt=args.prompt,
        references=args.reference,
        output_path=args.output,
        image_size=args.size,
        aspect_ratio=args.ratio,
    )
    print(
        json.dumps(
            {
                "provider": result.provider,
                "model": result.model,
                "output": str(args.output.expanduser().resolve()),
                "usage": result.usage,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
