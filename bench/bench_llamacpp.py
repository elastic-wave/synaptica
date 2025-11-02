# bench/bench_llamacpp.py
import argparse, time, csv, json, pathlib, subprocess, requests
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("models/TinyLlama_TinyLlama-1.1B-Chat-v1.0", use_fast=True)

LLAMA_URL = "http://127.0.0.1:8080/completion"

def read_prompts(path):
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

class TegraStats:
    def __init__(self, log_path):
        self.log_path = pathlib.Path(log_path)
        self.proc = None
    def __enter__(self):
        self.proc = subprocess.Popen([
            'tegrastats', '--interval', '1000'
        ], stdout=open(self.log_path, 'w'), stderr=subprocess.DEVNULL)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

def bench_prompt(prompt, n_predict=128, temperature=0.7, stream=True):
    headers = {'Content-Type': 'application/json'}
    payload = {
        'prompt': prompt,
        'n_predict': n_predict,
        'temperature': temperature,
        'stream': stream
    }
    t0 = time.time()
    ttfb = None
    text = []

    if stream:
        with requests.post(LLAMA_URL, data=json.dumps(payload), headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None):
                if not chunk:
                    continue
                if ttfb is None:
                    ttfb = (time.time() - t0) * 1000.0
                text.append(chunk.decode('utf-8', errors='ignore'))
        total_ms = (time.time() - t0) * 1000.0
        full = ''.join(text)
        return ttfb, total_ms, full
    else:
        r = requests.post(LLAMA_URL, data=json.dumps(payload), headers=headers, timeout=300)
        r.raise_for_status()
        ttfb = (time.time() - t0) * 1000.0
        data = r.json() if r.headers.get('Content-Type','').startswith('application/json') else {}
        content = data.get('content') if isinstance(data, dict) else r.text
        total_ms = (time.time() - t0) * 1000.0
        return ttfb, total_ms, content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_predict', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--no_stream', action='store_true')
    args = ap.parse_args()

    prompts = read_prompts(args.prompts)
    out_csv = pathlib.Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tegra_log = out_csv.with_suffix('.tegrastats.log')

    with TegraStats(tegra_log):
        rows = []
        for p in prompts:
            ttfb, total_ms, out_text = bench_prompt(
                p, n_predict=args.n_predict, temperature=args.temperature, stream=(not args.no_stream)
            )
            # very rough token estimate: split on whitespace
            est_tokens = len(TOKENIZER.encode(out_text))
            tok_per_s = est_tokens / (total_ms/1000.0) if total_ms > 0 else 0.0
            print(f"TTFB={ttfb:.1f} ms  total={total_ms:.1f} ms  est_tok/s={tok_per_s:.2f}")
            rows.append({
                'prompt': p,
                'ttfb_ms': f"{ttfb:.1f}",
                'total_ms': f"{total_ms:.1f}",
                'est_tokens': est_tokens,
                'est_tok_per_s': f"{tok_per_s:.2f}"
            })

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt','ttfb_ms','total_ms','est_tokens','est_tok_per_s'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[bench] Saved {out_csv} and {tegra_log}")

if __name__ == '__main__':
    main()