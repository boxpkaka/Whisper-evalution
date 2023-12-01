# Whisper-evalution

## Whisper性能测试

用于whisper的中、粤、英测试

数据格式为wav.scp, text



## 模型目录

$表示eval.sh中需传入的参数

```sh
└─$model_dir
   ├─model_type
   │  └─model_name
   │
   └─$model_type
      └─model_name_list[$model_index]
         └─pytorch_model.bin
         └─config.json


└─$dataset_dir       
   ├─dataset_name
   │
   └─dataset_list[$data_index] 
      └─wav.scp
      └─text
```


