
## 📌 bert_data_plot
  modeling_bert.py에서 저장한 tensor 데이터 시각화
  
## 📌 MPWnormfile
  - **📁tensor_files** : 8.8 fixed point format의 tensor데이터 (.pt형태)
  - **📁text_files** :  8.8 fixed point format의 tensor데이터 (.txt형태)
  - **csv2hex_print.py** : 분산의 inverse_square_root LUT 값이 csv파일이여서 이걸  0xFFFF 형태로 변환
  - **tensor2txtfile_print.py** : tensor_files -> text_files 값으로 변경
  - invsqrt_w_LUT.py : inverse square root(variance)에서 사용된 LUT에 맞는 분산값 확인하려고..(아예 랜덤값을 넣으면 분산이 LUT와 맞지않아서 동작을 하지 않기때문)
    
## 📌 vert_data_plot
  modeling_vit.py에서 저장한 tensor 데이터 시각화
  - **📁attention_map_png** : attention score & prob & (out)의 출력값을 matrix 형태로 시각화  
  - **📁layernorm_histogram_and_topk_png** : first&second layernorm의 입력값을 histogram확인 및 top-10개 데이터 plot  
  - **📁python** : 그래프 그리는 코드들
  - **📁tensor** : modeling_bert.py에서 torch.save해서 tensor 저장
  - **📁token_range_png** : layernorm 입력값 boxplot (별로 인듯..)
  
## 📌 transformers
### BERT (fixed point 8.8 format + custom operation )
- **📁invsqrt_csv_file** : NNLUT로 만든, LUT d,s,t값 (64=LUT 64개 , 32= LUT32개 , 그냥 inv~ = 24개)
- **custom_norm.py** : config값인 layernorm_method 받아서 'original' , 'custom_invsqrt_norm', 'dualpath_norm' 中 선택
- **custom_invsqrt.py** : invsqrt_csv_file의 csv파일 읽어서 LUT의 d,s,t 값으로 넣어줌  
     (주석처리는 csv파일로 읽는것 말고 self.register_buffer로 그냥 LUT값 텐서로 넣어줌. MPW verilog코드에는 24개 LUT인데 이거는 32개로 해놨길래 주석처리함)
- **custom_softmax_fixed.py** : config값인 soft_method 받아서 'original' , 'cordic', 'base2' 中 선택 (cordic은 오류남)
- **custom_configuration_bert.py** : BERTconfig에 softmax_method, layernorm_method 추가함
- **custom_activations.py** :  NNLUT로 만든, LUT16개 이용한 "CustomGELU" 추가
- **modeling_bert.py**
  + Activation & weight & bias 모두 fixed point 8.8로 변경 = ~~torch.round~~ torch.floor , torch.clip 이용
  + Import Custom_\*\*\*
  + (현재는 주석처리) NVIDIA Tools for Profiling Code in Python [참고](https://developer.nvidia.com/ko-kr/blog/nvidia-tools-extension-api-python-%eb%b0%8f-c-c%ec%97%90%ec%84%9c-%ec%bd%94%eb%93%9c%eb%a5%bc-%ed%94%84%eb%a1%9c%ed%8c%8c%ec%9d%bc%eb%a7%81%ed%95%98%ea%b8%b0-%ec%9c%84%ed%95%9c-%ec%a3%bc%ec%84%9d/)  
     import torch.cuda.nvtx as nvtx, nvtx.range_push("이름") , nvtx.range_pop()
  + (현재는 주석처리) torch.save로 tensor값 저장

- **run_glue.py** : argument로 layernorm,gelu,softmax method 받아서 config로 설정
- **run_glue_models.sh** , run_glue_optional.sh : 여러번 run_glue하기 위함  

<details>
<summary> BERT accuracy</summary>
<div>
  data = mrpc 
  
1. original : full-precision  
eval_accuracy           =     0.8775

2. original with fixed-point 8.8  
eval_accuracy           =     0.8725

2. custom_GELU , base2softmax , custom_invsqrt_norm  WITH fixed-point 8.8  
eval_accuracy           =      0.875

</div>
</details>

### ViT (*일단 지금은 fp32 , original operation*)
- **custom_norm.py** : 위 BERT내용과 동일, *일단 지금은 'original'만 사용*
- **custom_invsqrt.py** :위 BERT내용과 동일, *일단 지금은 self.register_buffer로 그냥 BERT용으로 만들었던 32개 LUT값 이용함*
- **custom_softmax_fixed.py** : 위 BERT내용과 동일, *일단 지금은 'original'만 사용*
- **custom_configuration_vit.py** : 위 BERT내용과 동일, ViTconfig에 softmax_method, layernorm_meth

- **modeling_vit.py**
  + *일단 지금은 fp32*
  + Import Custom_\*\*\*
  + torch.save로 tensor값 저장
  + class ViTSdpaSelfAttention 수정 : torch.nn.functional.scaled_dot_product_attention으로 연산하는것 대신 class ViTSelfAttention처럼 작성 (중간 attention값을 torch.save하기위해)

- **run_image_classification.py** : 위 BERT내용과 동일
- **run_image_classification_models.sh** : run_image_classification.py 실행할때 argument 수정하기 편하려고 만듦

---
**수정한 파일 위치**

transformers  
├── src  
│   └── transformer  
│       ├── models  
│       │   ├── bert ─ ① custom_\*\*\*py  ② modeling_bert.py  ③invsqrt_csv_file  
│       │   └── vit ─ ① custom_\*\*\*py ② modeling_vit.py  
│       ├── custom_activation.py  
│  
├── run_glue.py  
├── run_image_classification.py    
└── run_\*\*\*\.sh  // python 실행 여러 번 하는 것 한 번에 하려고, 혹은 argument 주는것 적어놓음 

## Reference
1. huggingface transformer : https://github.com/huggingface/transformers
2. NNLUT : https://arxiv.org/abs/2112.02191


