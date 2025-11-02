# synaptica
A pipeline for pruning and optimizing local LLMs for the Jetson Orin Nano


# Notes

# commands to build (warning takes an hour or so)

cd /media/ubuntu/ssd_drive/llama.cpp
python3 convert_hf_to_gguf.py --outfile tinyllama-f16.gguf --outtype f16   /media/ubuntu/ssd_drive/projects/synaptica/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

./build/bin/llama-quantize tinyllama-f16.gguf tinyllama-q4km.gguf Q4_K_M

# quantize to a smaller model (outputs to llama.cpp folder)
./build/bin/llama-quantize tinyllama-f16.gguf tinyllama-q4km.gguf Q4_K_M

# test the results

# basic   
#./build/bin/llama-server -m /media/ubuntu/ssd_drive/llama.cpp/tinyllama-q4km.gguf -c 1024 --host 0.0.0.0 --port 8080 --gpu-layers 999

#verbose
MODEL=/media/ubuntu/ssd_drive/llama.cpp/tinyllama-q4km.gguf

./build/bin/llama-server -m "$MODEL" -c 512 --port 8080 --host 0.0.0.0 --gpu-layers 32 --verbose 

# smaller, cpu only

./build/bin/llama-server -m "$MODEL" -c 256 --port 8080 --host 127.0.0.1 --gpu-layers 0 --verbose 

# then from otehr terminal:
# ssh into jetsons or run from jetsons

#simple
curl -s http://localhost:8080/completion -d '{"prompt":"Hello Jetson","n_predict":64}'

#detailed
curl -v -H 'Content-Type: application/json' -d '{"prompt":"Hello Jetson","n_predict":64,"temperature":0.7}' http://127.0.0.1:8080/completion

#bench marking

# Use the shell helper script run_matrix to loop through different configuration, starting a server, running the benchmark 
# script and saving results to csv. Then can compare teh csv files to see best parameters for performance

cd /media/ubuntu/ssd_drive/projects/synaptica
bash bench/run_matrix.sh


