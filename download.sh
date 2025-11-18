#!/bin/bash

# Set your Hugging Face credentials
export HF_TOKEN="your_token_here"
export HF_USERNAME="your_username_here"
export HF_ENDPOINT=https://hf-mirror.com

# Download evaluation datasets
cd data

./hfd.sh Auraithm/AIME2024 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh Auraithm/AIME2025 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh Auraithm/MATH500 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh Auraithm/GSM8K --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh Auraithm/OlympiadBench --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME

# Download training datasets
./hfd.sh Auraithm/Light-OpenR1Math-SFT --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh Auraithm/Light-MATH-RL --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
cd ..

# Download models
cd public

./hfd.sh JetLM/SDAR-8B-Chat --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
./hfd.sh OpenMOSS-Team/DiRL-8B-Instruct --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME

cd ..

echo "All datasets and models downloaded successfully!"

