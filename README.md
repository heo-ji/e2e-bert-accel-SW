
# Bert-base모델의 end-to-end FPGA 가속기 구현을 위한 SW 정확도 확인용 코드
- [ESoC Lab.] mobile project / MPW 과제
- huggingface의 Mobel/Bert의 코드 수정
  -   custom_*.py 코드의 추가
  -   modeling_bert.py , run_glue.py 의 수정

# SW spec. & GLUE task 정확도
<details>
<summary>Q-format SW Golden Spec.</summary>

> **Fixed-point formats**
> *  input      : signed 16-bit Q8.8
> *  acc        : signed 26-bit Q18.8
> *  acc2       : signed 34-bit Q26.8
> *  mean       : signed 16-bit Q8.8
> *  mean2      : signed 32-bit Q16.16
> *  var        : signed 24-bit Q8.16
> *  invsqrt    : signed 16-bit Q8.8
> *  norm       : signed 16-bit Q8.8
>

> **Precision Quantization**
> *  Qformat의 소수부
> *  torch.floor
> *  Values are floored after scaling to the target fractional width.

> **Range policy**
> * Qformat의 정수부(정확히는 Qformat으로 나타낼 수 있는 min,max)
> * torch.clipping
> * Values outside the target Q-format range **are saturated to the min/max** representable value.

> **Variance 연산**
> * var = mean(x^2) - mean(x)^2
> * var is stored as signed Q8.16.
> * **Negative var is not explicitly clamped to zero.
>   Only signed Q8.16 range saturation is applied.**

> **Inverse sqrt 연산**
> * invsqrt input = var + eps
> * eps = 2^-16, the minimum positive Q8.16 value.
> * invsqrt is computed using the same LUT approximation rule as the RTL.
> * 24개의 LUT사용 , NNLUT 사용


</details>


<details>
<summary> BERT SW 모델 </summary>
<div>
<pre><code>
# 1. 필요하면 설정
pip install evaluate
pip uninstall -y peft
mkdir -p ../../result/result_bert_base_original

# 2. SW(FP32) original 정확도 BERT-BASE/GLUE task
[MNLI] eval_accuracy: 0.8463
[QNLI] eval_accuracy: 0.9129
[QQP]  eval_accuracy: 0.9088
[RTE]  eval_loss: 0.7149
[SST-2] eval_accuracy: 0.9323
[STS-B] eval_combined_score: 0.8944
[CoLA] eval_matthews_correlation: 0.6024
[MRPC] eval_accuracy: 0.8627
    
# 3. SW(FP32) + layernorm만 HW(FXP8.8) 정확도 BERT-BASE/GLUE task
[MNLI] eval_accuracy: 0.8414
[QNLI] eval_accuracy: 0.9072
[QQP]  eval_accuracy: 0.9071
[RTE]  eval_accuracy: 0.7220
[SST-2] eval_accuracy: 0.9323
[STS-B] eval_combined_score: 0.8916
[CoLA] eval_matthews_correlation: 0.6179
[MRPC] eval_accuracy: 0.8529

# 4. full HW(FXP8.8) 정확도 BERT-BASE/GLUE task
[MNLI] eval_accuracy: 0.83739
[QNLI] eval_accuracy: 0.9081
[QQP]  eval_accuracy: 0.9074
[RTE]  eval_accuracy: 0.7220
[SST-2] eval_accuracy: 0.9300
[STS-B] eval_combined_score: 0.8916
[CoLA] eval_matthews_correlation: 0.615343(더 좋아져)
[MRPC] eval_accuracy: 0.8725(더 좋아져)
  </code></pre>

</div>
</details>
 
# 파일 설명
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
  
## 📌 transformers/src/model/
### BERT (fixed point 8.8 format + custom operation )
- **📁invsqrt_csv_file** : NNLUT로 만든, LUT d,s,t값 (64=LUT 64개 , 32= LUT32개 , 그냥 inv~ = 24개)
- **custom_norm.py** : config값인 layernorm_method 받아서 'original' , 'custom_invsqrt_norm', 'dualpath_norm' 中 선택
- **custom_invsqrt.py** : invsqrt_csv_file의 csv파일 읽어서 LUT의 d,s,t 값으로 넣어줌  
     (주석처리는 csv파일로 읽는것 말고 self.register_buffer로 그냥 LUT값 텐서로 넣어줬던거. 근데 24개가 아니라 32개로 해놨길래 주석처리함)
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
# **수정한 파일 위치**

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



# 📌./layernorm_sw_golden  파일 설명
> HW와 비교할 SW_reference trace 파일(txt) 생성을 위한 프로젝트

1. **fxp_fp_sw_golden.py**
   - `def qformat_value`: clipping & floor 수행
   - `def sw_float_golden`: 부동소수점 기준 골든 모델
   - `def sw_qformat_golden`: `custom_norm.py`의 `forward_fxp88`과 동일 로직
   - **Return:** 
     ```python
     return dict(acc=acc, acc2=acc2, mean=mean, mean2=mean2,
                 var=var, invsqrt=invsqrt, normalized=normalized)
     ```

2. **invsqrt_LUT_fp_vs_fxp_compare.py**
   - Inverse square root(var)의 LUT 결과가 0이 되지 않는 `var` 범위 분석
   - (사실 필요없었음)그냥 LUT 16진수 값을 가지고 판단해도 상관없었을듯



# 📌최종 sw 수정 사항
1. magic num 수정
: torch.clip(input_fx16, -2**7, 127.99609375) 에서
127.99609375-> (2**7 - 1/scale_factor_8) 형태로 수정

2. modeling_bert.py, custom_softmax_fixed.py 에서 
torch.round ->torch.floor로 수정
: HW에서 하위비트 버림 = torch.floor 이기때문

- [ ]  수정한 SW glue task 정확도 확인 필요
- [ ]  #[try]# 지우고 돌려보기 정수부 clip 한것임. 없어도 정확도 상관없는거 같긴함.
