python -m eval.eval \
       --model_dir /data1/yumingdong/model/ \
       --model_type finetuned \
       --model_index 6 \
       --dataset_dir /data2/yumingdong/data \
       --data_index 7 \
       --export_dir /data1/yumingdong/whisper/whisper-eval/exp \
       --batch_size 16 \
       --language zh \
       --gpu 0
