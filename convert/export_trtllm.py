import pathlib, shutil
out_dir = pathlib.Path('build-input/tinyllama')
out_dir.mkdir(parents=True, exist_ok=True)
# simulate export: copy config/tokenizer/model files from models/... to out_dir
shutil.copytree('models/TinyLlama_TinyLlama-1.1B-Chat-v1.0', out_dir, dirs_exist_ok=True)
print('Export complete:', out_dir)
