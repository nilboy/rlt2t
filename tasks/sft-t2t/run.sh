kfold_id=$4
params=(
      --model_name_or_path /root/autodl-tmp/output-models/pretrain-t2t \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 1000 \
      --train_file data/t2t/$kfold_id/train.json \
      --validation_file data/t2t/$kfold_id/test.json \
      --text_column text \
      --summary_column summary \
      --source_prefix "" \
      --output_dir /root/autodl-tmp/output-models/sft-t2t-$kfold_id \
      --text_map_start_idx 106 \
      --text_map_num_words 1800 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 2 \
      --per_device_eval_batch_size 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_steps 20000 \
      --warmup_steps 1000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 1000 \
      --save_total_limit 1 \
      --predict_with_generate \
      --bf16
)

if [[ "$1" == "deepspeed" ]]; then
  deepspeed tasks/sft-t2t/run_t2t.py --deepspeed tasks/sft-t2t/ds_config.json "${params[@]}"
else
  python -m torch.distributed.launch --nproc_per_node "$2" --master_port="$3" tasks/sft-t2t/run_t2t.py "${params[@]}"
fi
