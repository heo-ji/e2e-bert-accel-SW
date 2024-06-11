import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# 디렉토리 설정
input_dir = "/home/user/HJH/2024_nonlinear/transformers/src/transformers/models/vit/vit_input_range"
files = sorted(os.listdir(input_dir))





# 각 파일을 읽고 boxplot 생성
for file in files:
    if file.endswith(".pt"):
        file_path = os.path.join(input_dir, file)
        hidden_states = torch.load(file_path)

        # 첫 번째 배치의 hidden states 선택
        batch1 = hidden_states[0].cpu().numpy()

        
        # ranges = batch1.max(axis=1) - batch1.min(axis=1)

        # Boxplot 생성
        plt.figure(figsize=(70, 6))
        plt.boxplot(batch1.T, vert=True, patch_artist=True, whis=2)  # batch1.T는 [768, 129]로 변환
        plt.xlabel('Patch Index')
        plt.ylabel('Range')
        plt.title(file)
        plt.grid(True)

        # 그래프 저장
        plot_file_path = os.path.join(input_dir, f"{file}.png")
        plt.savefig(plot_file_path)
        plt.close()

