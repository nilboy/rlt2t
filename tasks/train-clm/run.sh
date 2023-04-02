params=(
      --model_name_or_path /root/autodl-tmp/pmodels/Wenzhong2.0-GPT2-3.5B-chinese \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 2000 \
      --train_file data/clm/train.json \
      --validation_file data/clm/test.json \
      --output_dir /root/autodl-tmp/output-models/clm \
      --max_seq_length 256 \
      --text_map_start_idx 106 \
      --text_map_num_words 1800 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --per_device_eval_batch_size 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_steps 100000 \
      --warmup_steps 5000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 2000 \
      --save_total_limit 1 \
      --bf16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed --master_port="$2" tasks/train-clm/train_clm.py --deepspeed tasks/train-clm/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" --master_port="$3" tasks/train-clm/train_clm.py "${params[@]}"
fi
