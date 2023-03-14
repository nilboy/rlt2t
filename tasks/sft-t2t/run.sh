deepspeed --num_gpus=1 tasks/sft-t2t/run_t2t.py \
      --deepspeed tasks/sft-t2t/ds_config.json \
      --model_name_or_path /root/autodl-tmp/models/IDEA-CCNL/Randeng-T5-77M \
      --do_train \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 1000 \
      --train_file /root/autodl-tmp/data/train.json \
      --validation_file /root/autodl-tmp/data/test.json \
      --text_column text \
      --summary_column summary \
      --source_prefix "" \
      --output_dir /root/autodl-tmp/output-models/sft-t2t \
      --text_map_start_idx 106 \
      --text_map_num_words 7000 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 8 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --max_eval_samples 100 \
      --max_steps 40000 \
      --warmup_steps 1000 \
      --lr_scheduler_type linear \
      --logging_steps 20 \
      --save_steps 1000 \
      --save_total_limit 2 \
      --predict_with_generate \
      --bf16
