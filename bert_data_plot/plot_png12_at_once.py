# name= pltname[5] ##0~5 변경


import matplotlib.pyplot as plt
from PIL import Image
import os

# 이미지 파일이 위치한 디렉토리
path = './'  # 이미지 파일 경로를 적절히 수정하세요.

# 파일명 패턴에 따른 이미지 파일명 리스트 생성
pltname = ["_Attention_layernorm_histogram", 
        "_Attention_layernorm_topk",
        "_FFN_topk","_FFN_histogram",
        "_FFN_layernorm_histogram", "_FFN_layernorm_topk"]

name= pltname[5] ##0~5 변경


file_names = [f"{i}{name}.png" for i in range(1, 13)]  # 1부터 12까지 예시

# 이미지 표시
fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # 3행 4열의 서브플롯 생성
axs = axs.ravel()  # 다차원 배열을 1차원으로 평탄화

for i, file_name in enumerate(file_names):
    # 이미지 파일 로드
    img_path = os.path.join(path, file_name)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].set_title(file_name)
        axs[i].axis('off')  # 축을 표시하지 않음
    else:
        axs[i].set_visible(False)  # 파일이 없는 경우 해당 subplot을 숨김

plt.tight_layout()
plt.show()

# 전체 그림을 파일로 저장
plt.savefig(f"{name}")


plt.close()