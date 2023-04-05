params=(
      --model_name_or_path /root/autodl-tmp/pmodels/roberta-base \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 2000 \
      --train_file data/mlm/train.json \
      --validation_file data/mlm/test.json \
      --output_dir /root/autodl-tmp/output-models/pretrain-mlm \
      --max_seq_length 256 \
      --line_by_line true \
      --text_map_start_idx 106 \
      --text_map_num_words 1800 \
      --per_device_train_batch_size 128 \
      --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 16 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_steps 100000 \
      --warmup_steps 5000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 2000 \
      --save_total_limit 1 \
      --fp16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed tasks/pretrain-mlm/pretrain_mlm.py --deepspeed tasks/pretrain-mlm/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" --master_port="$3" tasks/pretrain-mlm/pretrain_mlm.py "${params[@]}"
fi
