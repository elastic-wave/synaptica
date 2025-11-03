#first time setup
cd projects/synaptica
python3 -m venv .venv
source .venv/bin/activate

#run




#on the nano
ssh ubuntu@jetsons
cd projects/synaptica
source .venv/bin/activate

docker build -t trtllm-jetson -f build/Dockerfile.jetson .



