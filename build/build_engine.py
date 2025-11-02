# build/build_engine.py
import argparse, json, subprocess, pathlib, time, itertools

def run(cmd):
    print("[build]", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    args = ap.parse_args()

    recipe = json.loads(pathlib.Path(args.recipe).read_text())
    model_id = recipe["model_id"]
    safe = model_id.replace("/", "_")
    ckpt = pathlib.Path("build-input/tinyllama")  # matches your export step

    ctx_list = recipe["ctx_lengths"]
    wp_list  = recipe["weights_precision"]
    kv_list  = recipe["kv_cache_precision"]
    pkv_list = recipe.get("paged_kv_cache", [True])
    mb_list  = recipe["max_batch"]
    psize    = recipe.get("page_size", [128])
    attn_plg = recipe.get("use_gpt_attention_plugin", [True])
    gemm     = recipe.get("use_gemm_plugin", ["auto"])
    inflight = recipe.get("enable_inflight_batching", [True])
    cgraph   = recipe.get("use_cuda_graph", [True])

    for ctx, wp, kv, pkv, mb, pg, attn, gp, ib, cg in itertools.product(
        ctx_list, wp_list, kv_list, pkv_list, mb_list, psize, attn_plg, gemm, inflight, cgraph
    ):
        name = f"{safe}-ctx{ctx}-{wp}-kv{kv}-mb{mb}-pg{pg}"
        out_dir = pathlib.Path("releases") / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "trtllm-build",
            f"--checkpoint_dir={ckpt}",
            f"--output_dir={out_dir}",
            f"--max_batch_size={mb}",
            f"--max_input_len={ctx}",
        ]

        # Precision/quant flags (names vary by TRT-LLM version)
        if wp == "fp16":
            cmd += ["--fp16"]
        elif wp == "int8_wo4":
            cmd += ["--int8"]          # weight-only INT8 base
            cmd += ["--weight_only"]   # and allow 4-bit where supported (hybrid)
        # KV cache
        if kv == "int8":
            cmd += ["--int8_kv_cache"]

        # Perf features
        if pkv:   cmd += ["--paged_kv_cache", f"--tokens_per_block={pg}"]
        if attn:  cmd += ["--use_gpt_attention_plugin"]
        if gp:    cmd += [f"--gemm_plugin={gp}"]
        if ib:    cmd += ["--enable_inflight_batching"]
        if cg:    cmd += ["--use_cuda_graph"]

        # (Optional) point to calibration set for INT8 PTQ:
        # cmd += ["--calib_data_dir=calib/shards"]

        run(cmd)

        (out_dir / "MANIFEST.json").write_text(json.dumps({
            "model_id": model_id,
            "ctx": ctx, "weights_precision": wp, "kv_cache_precision": kv,
            "paged_kv_cache": pkv, "page_size": pg, "max_batch": mb,
            "attention_plugin": attn, "gemm_plugin": gp,
            "inflight_batching": ib, "cuda_graph": cg,
            "timestamp": int(time.time())
        }, indent=2))

if __name__ == "__main__":
    main()
