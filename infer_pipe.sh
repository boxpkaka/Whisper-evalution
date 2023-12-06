cd /data1/yumingdong/whisper/whisper-eval/ || exit
python -m infer.infer_pipe \
--audio_path $1 \
--model_path $2 \
--language $3 \
--gpu $4 \
