# 배열 (128, 768)


import numpy as np
import torch

path = "/HJH/tensor/bert-base-cased/mrpc/tensordata/"
file = [
    "11_FFN_layernorm_input.pt",
    "11_FFN_layernorm_output.pt"]

# file = [
#     "11_attention_layernorm_input.pt",
#     "11_attention_layernorm_output.pt"]

def get_tensor2numpy(path, file, in_or_out=True):
  device = torch.device('cpu')

  input_tensor = torch.load(f"{path}{file[0]}", map_location=device) #load input fp32
  input_tensor = input_tensor[0] #batch = 1 's data'

  output_tensor = torch.load(f"{path}{file[1]}", map_location=device) #load output fp32
  output_tensor = output_tensor[0]  #batch = 1 's data


  input_data = input_tensor.numpy() #(128, 768) data[0]~data[127]행
  output_data = output_tensor.numpy() #(128, 768) data[0]~data[127]행

  if in_or_out :
    return input_data
  else :
    return output_data

def float_to_fixed_point(data, fractional_bits=8, full_bit=16):  #-128~127.99..  #int16(-32768 ~ 32767) 
    # 스케일링 인자 계산
    scale_factor = 2**fractional_bits
    
    # 데이터를 고정 소수점 표현으로 변환
    fixed_point_data = np.round(data * scale_factor)
    
    # 데이터를 16bit 범위로(-32768 ~ 32767) 제한
    fixed_point_data = np.clip(fixed_point_data, -2**(full_bit - 1), 2**(full_bit - 1) - 1)
    
    return fixed_point_data.astype(np.int16)

def fixed_to_real(data ,fractional_bits=8, full_bit=16):
    # 스케일링 인자 계산
    scale_factor = 2**fractional_bits
    
    # 부호 비트를 확인하고, 데이터의 범위를 -32768 ~ 32767로 조정
    # uint16 형태로 저장된 값이므로, int16 범위의 값으로 변환해줘야 합니다.
    data_signed = data.astype(np.int16)
    
    # 고정 소수점 데이터를 부동 소수점 데이터로 변환
    float_data = data_signed / scale_factor
    
    return float_data

#
inputdata = get_tensor2numpy(path, file, True)  #data.shape = (128, 768)
#fp32 연산
fp_means = np.mean(inputdata, axis=1)
fp_vars = np.var(inputdata, axis=1)
fp_stds = np.sqrt(fp_vars)
fp_normalized = (inputdata - fp_means[:, np.newaxis]) / fp_stds[:, np.newaxis]

bit16_in = float_to_fixed_point(inputdata) 

tokens = bit16_in.shape[0] #128
dmodel = bit16_in.shape[1] #768
dk=64
head= dmodel/dk


row_sum = np.zeros(tokens).astype(np.int32)
row_means = np.zeros(tokens).astype(np.int32)
row_pasum = np.zeros(tokens).astype(np.int64)
row_var = np.zeros(tokens).astype(np.int32)
row_std = np.zeros(tokens) .astype(np.int32)

for i in range(tokens):
  row_sum[i]= 0
  row_means[i] = 0
  for j in range(dmodel):
    row_sum[i] += bit16_in[i][j]  #adder 22bit for dk , 
  
  #row_means[i]=row_sum[i]/dmodel            
  row_means[i]=(row_sum[i]>>8)*0.33203125
  
INT32_MAX = 2**31 - 1
for i in range(tokens):
  row_pasum[i]= 0
  row_var[i] = 0
  row_std[i] =0
  for j in range(dmodel):
    row_pasum[i] += ((bit16_in[i][j] - row_means[i])**2)
    row_pasum[i] = min(row_pasum[i], INT32_MAX)

  #row_var[i]=row_pasum[i]/dmodel             
  row_var[i]=(row_pasum[i]>>8)*0.33203125
  row_std[i] = np.sqrt(row_var[i])



normalized = ((bit16_in) - (row_means[:, np.newaxis])) / (row_std[:, np.newaxis])
# breakpoint()
print("차이 mean = ",np.mean(np.abs((fp_normalized - normalized)) ))
# breakpoint()

# with open('FFN_layernorm_data.txt', 'w') as file:
#     print('fp32 input_data : ',np.min(inputdata), '~' ,np.max(inputdata),'  int16 input_data : ',np.min(bit16_in), '~' ,np.max(bit16_in), file=file)
#     print('fp_means : ',np.min(fp_means), '~' ,np.max(fp_means),'\n', file=file)
#     print('fp_vars : ',np.min(fp_vars), '~' ,np.max(fp_vars),'\n', file=file)
#     print('fp_stds : ',np.min(fp_stds), '~' ,np.max(fp_stds),'\n', file=file)
#     print('fp_normalized : ',np.min(fp_normalized), '~' ,np.max(fp_normalized),file=file)

#     print('int64 row_sum : ',np.min(row_sum), '~' ,np.max(row_sum), file=file)
#     print('int64 row_means : ',np.min(row_means), '~' ,np.max(row_means), file=file)
#     print('int64 row_pasum : ',np.min(row_pasum), '~' ,np.max(row_pasum), file=file)
#     print('int64 row_var : ',np.min(row_var), '~' ,np.max(row_var),'\n', file=file)
#     print('int64 row_std : ',np.min(row_std), '~' ,np.max(row_std),'\n', file=file)
#     print('int_normalized : ',np.min(normalized), '~' ,np.max(normalized),'\n', file=file)

np.set_printoptions(threshold=np.inf)#, suppress=True)

with open('11_FFN_layernorm_inputdata_int_768.txt', 'w') as file:
   #print(bit16_in.flatten(),file=file)
   print(bit16_in[0,:],file=file)


# token=1에서 16개만
with open('11_FFN_layernorm_inputdata_int_16.txt', 'w') as file:
  #  print(fp_normalized.flatten(),file=file)
   print(bit16_in[0,:16],file=file)

with open('11_FFN_layernorm_normout_fp_16.txt', 'w') as file:
    print(fp_normalized[0,:16],'\n',file=file)

#     print('int64 row_sum : ', row_sum,'\n', file=file)
#     print('int64 row_means : ',row_means,'\n', file=file)
#     print('int64 row_pasum : ',row_pasum,'\n', file=file)
#     print('int64 row_var : ',row_var,'\n', file=file)
#     print('int64 row_std : ',row_std,'\n', file=file)
#     print('int_normalized : ',normalized[0,:16],'\n', file=file)






