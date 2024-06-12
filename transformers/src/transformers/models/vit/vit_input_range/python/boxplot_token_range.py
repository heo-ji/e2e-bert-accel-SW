import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

# 디렉토리 설정
common_dir = "/home/user/HJH/2024_nonlinear/transformers/src/transformers/models/vit/vit_input_range"
# data
data_name = "patch16-224-cifar10"

input_dir = os.path.join(common_dir, f"./tensor/{data_name}")
files = sorted(os.listdir(input_dir))

# first_layernorm과 second_layernorm 파일을 분리
# 파일 목록을 정렬할 때 숫자 부분을 기준으로 정렬!
first_layernorm_files = sorted([file for file in files if 'first_layernorm' in file], key=lambda x: int(re.findall(r'\d+', x)[0]))
second_layernorm_files = sorted([file for file in files if 'second_layernorm' in file], key=lambda x: int(re.findall(r'\d+', x)[0]))

# 각 파일을 읽고 boxplot 생성
def create_boxplots(layernorm_files, layer_type):
    fig, axs = plt.subplots(3, 4, figsize=(24, 18), tight_layout=True)
    fig.suptitle(f'{layer_type.capitalize()} Layernorm Boxplots', fontsize=16)

    for i, file in enumerate(layernorm_files):
        file_path = os.path.join(input_dir, file)
        hidden_states = torch.load(file_path)

        # 첫 번째 배치의 hidden states 선택
        batch1 = hidden_states[0].cpu().numpy()

        row, col = divmod(i, 4)
        axs[row, col].boxplot(batch1.T, vert=True, patch_artist=True, whis=2)  # batch1.T는 [768, 129]로 변환
        axs[row, col].set_xlabel('Patch Index')
        axs[row, col].set_ylabel('Range')
        axs[row, col].set_title(file)
        axs[row, col].grid(True)

    # 그래프 저장
    plot_file_path = os.path.join(common_dir, f"./token_range_png/{layer_type}_layernorm_boxplots.png")
    plt.savefig(plot_file_path)
    plt.show()
    plt.close()

# first_layernorm과 second_layernorm 각각에 대해 boxplot 생성
create_boxplots(first_layernorm_files, 'first')
create_boxplots(second_layernorm_files, 'second')
