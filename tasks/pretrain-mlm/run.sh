params=(
      --model_name_or_path /root/autodl-tmp/models/IDEA-CCNL/Erlangshen-MegatronBert-1.3B \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 1000 \
      --train_file /root/autodl-tmp/data/train.json \
      --validation_file /root/autodl-tmp/data/test.json \
      --output_dir /root/autodl-tmp/output-models/pretrain-mlm-bf16 \
      --max_seq_length 512 \
      --line_by_line true \
      --text_map_start_idx 106 \
      --text_map_num_words 7000 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 4 \
      --per_device_eval_batch_size 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_steps 100000 \
      --warmup_steps 1000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 500 \
      --save_total_limit 2 \
      --bf16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed tasks/pretrain-mlm/pretrain_mlm.py --deepspeed tasks/pretrain-mlm/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" tasks/pretrain-mlm/pretrain_mlm.py "${params[@]}"
fi
