params=(
      --model_name_or_path /root/autodl-tmp/models/IDEA-CCNL/Randeng-T5-77M \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 1000 \
      --train_file /root/autodl-tmp/data/pretrain-t2t.json \
      --validation_file /root/autodl-tmp/data/pretrain-t2t-test.json \
      --text_column text \
      --output_dir /root/autodl-tmp/output-models/pretrain-t2t \
      --text_map_start_idx 106 \
      --text_map_num_words 7000 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 4 \
      --per_device_eval_batch_size 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_steps 100000 \
      --warmup_steps 1000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 1000 \
      --save_total_limit 2 \
      --predict_with_generate \
      --bf16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed tasks/pretrain-t2t/pretrain_t2t.py --deepspeed tasks/pretrain-t2t/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" tasks/pretrain-t2t/pretrain_t2t.py "${params[@]}"
fi
