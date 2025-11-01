from huggingface_hub import snapshot_download
import hashlib, json, pathlib

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
out_dir = pathlib.Path('models') / model_id.replace('/', '_')
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {model_id} to {out_dir}...")
local_dir = snapshot_download(repo_id=model_id, local_dir=out_dir)

# Build a manifest of file hashes
manifest = {}
for path in pathlib.Path(local_dir).rglob('*'):
    if path.is_file():
        with open(path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        manifest[str(path.relative_to(local_dir))] = h

manifest_path = out_dir / 'MANIFEST.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Manifest written to {manifest_path}")
