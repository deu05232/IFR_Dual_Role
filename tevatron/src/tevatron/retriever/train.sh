
# Promptriever
nohup bash -c 'deepspeed --include localhost:"0,1" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir "promptriever-llama2-7B-new_seed42" \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 100 \
  --dataset_name "DATASET" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 32 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --save_total_limit 2 \
  --data_seed 42 \
  --dont_shuffle \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --negatives_first_n 3' > logs/train.log 2>&1 &


# JointLH
nohup bash -c 'deepspeed --include localhost:"0,1" --master_port "60001" train_multi.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir "promptriever-llama2-7B-new_seed42-JointLH" \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 100 \
  --dataset_name "DATASET" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 32 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --save_total_limit 2 \
  --data_seed 42 \
  --dont_shuffle \
  --num_positives 2 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --negatives_first_n 3' > logs/train.log 2>&1 &



# RandLH
nohup bash -c 'deepspeed --include localhost:"0,1" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir "promptriever-llama2-7B-new_seed42-RandLH" \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 100 \
  --dataset_name "DATASET" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 32 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --save_total_limit 2 \
  --data_seed 42 \
  --dont_shuffle \
  --gradient_accumulation_steps 2 \
  --negatives_first_n 3' > logs/train.log 2>&1 &

# add_sample
nohup bash -c 'deepspeed --include localhost:"0,1" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir "promptriever-llama2-7B-add_all_q" \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 100 \
  --dataset_name "DATASET" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 32 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --save_total_limit 2 \
  --data_seed 42 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --dont_shuffle \
  --negatives_first_n 3' > logs/train.log 2>&1 &