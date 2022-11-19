python3 get_data.py

python3 run_seq2seq_1B.py \
    --model_name_or_path AndrewZeng/S2KG-base \
    --tokenizer_name AndrewZeng/S2KG-base \
    --model_name t5 \
    --do_eval \
    --train_file ./data/test.json \
    --validation_file ./data/test.json \
    --source_prefix "dialogue: " \
    --output_dir ./result \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
	--eval_steps=50 \
    --logging_steps=10 \
    --text_column dialogue \
    --summary_column response \
    --num_train_epochs=20.0 \
    --learning_rate=2e-5 \
    --warmup_ratio=0.1 \
    --num_beams=7 \