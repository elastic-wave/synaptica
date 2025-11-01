# build/build_engine.py
import argparse, json, hashlib, os, pathlib, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    args = ap.parse_args()

    with open(args.recipe) as f:
        recipe = json.load(f)
    model_id = recipe["model_id"]
    safe_name = model_id.replace("/", "_")

    out_root = pathlib.Path("releases")
    out_root.mkdir(exist_ok=True)

    # Simulate two builds: fp16 and int8
    for prec in ["fp16", "int8"]:
        out_dir = out_root / f"{safe_name}-{prec}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy engine file to prove the pipeline
        engine_path = out_dir / "model.engine"
        with open(engine_path, "wb") as f:
            f.write(b"DUMMY_TRT_ENGINE_PLACEHOLDER")

        # Minimal manifest
        manifest = {
            "model_id": model_id,
            "precision": prec,
            "ctx_lengths": recipe.get("ctx_lengths", []),
            "kv_cache_precision": recipe.get("kv_cache_precision", []),
            "paged_kv_cache": recipe.get("paged_kv_cache", []),
            "max_batch": recipe.get("max_batch", []),
            "timestamp": int(time.time())
        }
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

        # Hash
        h = hashlib.sha256(engine_path.read_bytes()).hexdigest()
        (out_dir / "HASH.txt").write_text(h + "\n")

        print(f"[build] Wrote dummy engine: {engine_path}")

    print("[build] Placeholder build complete. Replace this script’s middle section with real TRT-LLM calls when ready.")

if __name__ == "__main__":
    main()
