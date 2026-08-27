# GRS.AI Nano Banana 2 smoke experiment

This experiment calls only `grsai.images.generate`. It does not select or
fall back to the official Gemini provider.

Configure `GRSAI_API_KEY` in the normal ai-toolkit environment source, then
run from the repository root:

```bash
.venv/bin/python experiments/grsai_nano_banana_2/run.py \
  --prompt "A clean pixel-art farmer on a flat magenta background." \
  --size 1K \
  --ratio 1:1
```

Use `--reference path/to/image.png` once per reference image. Generated files
are written under this experiment's ignored `outputs/` directory unless an
explicit `--output` path is provided.
