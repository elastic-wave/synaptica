# convert/export_trtllm.py
import argparse, pathlib, shutil
from transformers import AutoConfig, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)         # e.g., build-input/tinyllama
    args = ap.parse_args()

    src = pathlib.Path("models") / args.model_id.replace("/", "_")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Basic sanity: keep HF files + tokenizer in one place for TRT-LLM
    cfg = AutoConfig.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src, use_fast=True)
    tok.save_pretrained(out)

    # For TinyLlama (LLaMA arch), TRT-LLM has a "convert_checkpoint" utility in python
    # Fallback: copy raw HF files; the builder will read from this directory.
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, out / p.name)

    print("TRT-LLM export staging complete:", out)

if __name__ == "__main__":
    main()
