#!/usr/bin/bash

# Set working directory
cd ~/nnscaler/examples/llama3_8B_128K

# Create logs directory if it doesn't exist
mkdir -p /blob/logs/nnscaler/llama_8B_128K_cuda_4

# Execute the first command and redirect output to the data preparation log
echo "Running data preparation..."
python bookcorpus.py --data_path_or_name bookcorpus/bookcorpus \
                    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
                    --save_path /scratch/nnscaler/bookcorpus_llama3_4K \
                    --sequence_length 4096 \
                    > /blob/logs/nnscaler/llama_8B_128K_cuda_4/data_prepare.log 2>&1

# Execute the second command
echo "Creating mini model..."
python create_mini_model.py --model_id meta-llama/Meta-Llama-3-8B-Instruct \
                            --output_id /scratch/nnscaler/llama3_mini

# Execute the third command and redirect output to the training log
echo "Starting training..."
torchrun --nproc_per_node=$(echo $WORLD_SIZE) \
         train.py --plan_ngpus 1 \
                  --runtime_ngpus $(echo $WORLD_SIZE) \
                  --name llama3_debug \
                  --model_id /scratch/nnscaler/llama3_mini \
                  --dataset_path /scratch/nnscaler/bookcorpus_llama3_4K -p \
                  > /blob/logs/nnscaler/llama_8B_128K_cuda_4/train.log 2>&1
