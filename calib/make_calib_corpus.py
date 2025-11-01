# calib/make_calib_corpus.py
import pathlib, json
out_dir = pathlib.Path('calib/shards')
out_dir.mkdir(parents=True, exist_ok=True)

# create a small dummy corpus file
sample = ["This is a calibration sample.", "Another small text chunk."]
with open(out_dir / 'shard_000.json', 'w') as f:
    json.dump(sample, f)

print('Created calibration shard in', out_dir)
