params=(
      --model_name_or_path /root/autodl-tmp/pretrained-models/opt-125m \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 1000 \
      --train_file /root/autodl-tmp/data/train.json \
      --validation_file /root/autodl-tmp/data/test.json \
      --output_dir /root/autodl-tmp/output-models/train-clm-bf16 \
      --max_seq_length 2048 \
      --text_map_start_idx 106 \
      --text_map_num_words 7000 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 8 \
      --learning_rate 1e-4 \
      --weight_decay 0.0 \
      --max_steps 100000 \
      --warmup_steps 1000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 1000 \
      --save_total_limit 2 \
      --bf16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed tasks/train-clm/train_clm.py --deepspeed tasks/train-clm/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" tasks/train-clm/train_clm.py "${params[@]}"
fi
