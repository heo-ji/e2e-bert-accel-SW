##########################5_
#tensor보다 #numpy로 plt하는게 빠름
# for (batch = 8 )
#     for (layer 1~12)
#         save.tensor =[8,  . , ]을 저장 시키고 -> layernum_파일명.pt로 수정
#     batch 한번만 돌렸음. 따라서 tensor[0]의 데이터만 유효함

## layer 변경 : 10_ 변경
###################

import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch


path = "./bert-base-cased/mrpc/tensordata/"


file = [
    "10_attention_layernorm_input",
    "10_attention_layernorm_output"]
hist_ranges = [(-5, 5), (-5, 5)]

device = torch.device('cpu')

for i, file_name in enumerate(file):
    full_filename = f"{path}{file_name}.pt"
    tensor = torch.load(full_filename, map_location=device)
    tensor = tensor[0] ##첫번째 배치값만 유효한 값임

    data = tensor.numpy()
    data_flattened = data.flatten()
    
    # 서브플롯 생성
    plt.subplot(1, 2, i + 1)
    plt.hist(data_flattened, bins=100, alpha=0.75, range=hist_ranges[i])
    plt.title(f"{file_name}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')

# 전체 그림을 파일로 저장
plt.savefig("./10_Attention_layernorm_histogram.png")

# 현재 플롯 클리어
plt.show()
plt.close()


# 파일별로 처리
for i, file_name in enumerate(file):
    # 전체 파일명 생성 및 텐서 로드
    full_filename = f"{path}{file_name}.pt"
    tensor = torch.load(full_filename, map_location=device)
    tensor = tensor[0] ##첫번째 배치값만 유효한 값임
    
    # 상위 10개 값을 찾음 (torch.topk 사용)
    topk_values = torch.topk(tensor, k=10, dim=0).values.numpy()
    
    # Subplot 생성
    plt.subplot(1, 2, i + 1)
    
    # 열 인덱스 설정
    x = np.arange(tensor.size(1))  # 열의 개수 (3072)
    
    # 상위 값 개수 (10)에 대해 scatter plot 생성
    for j in range(topk_values.shape[0]):  # 상위 값 개수 (10)
        y = topk_values[j, :]
        plt.scatter(x, y, marker='.')
    plt.title(f'{file_name}', fontsize=10)
    plt.xlabel('Column Index', fontsize=10)
    plt.ylabel('Top-10 Values', fontsize=10)
    

# 전체 그림 저장

plt.savefig("./10_Attention_layernorm_topk.png")

# 화면에 보여주기
plt.show()

# 현재 플롯 클리어
plt.close()
