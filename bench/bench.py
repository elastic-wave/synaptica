import argparse, time, csv, subprocess, pathlib

def read_prompts(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def start_tegrastats(log_path):
    return subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=open(log_path, 'w'), stderr=subprocess.DEVNULL)

def stop_tegrastats(proc):
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

def run_bench(engine, prompts, out_csv):
    out_path = pathlib.Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Start tegrastats logging
    tegra_log = out_path.with_suffix('.tegrastats.log')
    tegra_proc = start_tegrastats(tegra_log)
    print(f"[bench] Logging tegrastats to {tegra_log}")

    results = []
    for p in prompts:
        t0 = time.time()
        # Dummy inference call (replace with actual later)
        time.sleep(0.5)  # simulate latency
        t1 = time.time()
        latency = round((t1 - t0) * 1000, 2)
        print(f"Prompt: {p[:40]}... | Latency: {latency} ms")
        results.append({'prompt': p, 'latency_ms': latency})

    stop_tegrastats(tegra_proc)

    # Write CSV results
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'latency_ms'])
        writer.writeheader()
        writer.writerows(results)

    print(f"[bench] Results saved to {out_csv}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True, help='Path to model.engine')
    ap.add_argument('--prompts', required=True, help='Path to prompts text file')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()

    prompts = read_prompts(args.prompts)
    run_bench(args.engine, prompts, args.out)