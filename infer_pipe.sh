python /data1/yumingdong/whisper/whisper-eval/infer/infer_pipe.py \
--audio_path $1 \
--model_path $2 \
--language $3 \
--gpu $4 \
--use_flash_attention_2 True \
--use_bettertransformer True