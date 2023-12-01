python -m eval.eval_whisper \
       --model_dir /data1/yumingdong/model/ \
       --model_type huggingface \
       --model_index 0 \
       --dataset_dir /data2/yumingdong/data \
       --data_index 4 \
       --export_dir /data1/yumingdong/whisper/whisper-eval/exp \
       --batch_size 16 \
       --language yue \
       --gpu 6
