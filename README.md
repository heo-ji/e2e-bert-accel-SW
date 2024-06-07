# BERT
- custom layernorm
- custom gelu
- custom softmax

run_glue.py의 argument이용해, config로 method 설정할 수 있도록함

# ViT
- custom layernorm
- custom softmax

run_image_classification.py에 마찬가지로 수정


**수정한 파일**

transformer  
├── src  
│   └── transformer  
│       ├── models  
│       │   ├── bert ─ custom_\*\*\*py , modeling_bert.py  
│       │   └── vit ─ custom_\*\*\*py , modeling_vit.py  
│       ├── custom_activation.py // BERT에서 LUT16개 이용한 "CustomGELU" 사용하려고 만듦  
│  
├── run_glue.py  
├── run_image_classification.py  
└── run_\*\*\*\.sh  // python 실행 여러 번 하는 것 한 번에 하려고  


