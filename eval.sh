python -m eval.eval \
       --model_dir /data1/yumingdong/model/ \
       --model_type ct2 \
       --model_index 5 \
       --dataset_dir /data2/yumingdong/data \
       --data_index 0 \
       --export_dir /data1/yumingdong/whisper/whisper-eval/exp \
       --batch_size 16 \
       --language zh \
       --num_workers 16 \
       --gpu 7 \
#       --int8 1 \
#       --pipeline 1
