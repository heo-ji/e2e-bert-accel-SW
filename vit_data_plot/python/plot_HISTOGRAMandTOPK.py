'''
BATCH [0]

'''

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch

# 디렉토리 설정
common_dir = "/home/user/HJH/2024_git_nonlinear/vit_data_plot"
# data
data_name = "patch16-224-cifar10"

input_dir = os.path.join(common_dir, f"./tensor/{data_name}")
files = sorted(os.listdir(input_dir))

# first_layernorm과 second_layernorm 파일을 분리
# 파일 목록을 정렬할 때 숫자 부분을 기준으로 정렬!
first_layernorm_files = sorted([file for file in files if 'first_layernorm' in file], key=lambda x: int(re.findall(r'\d+', x)[0]))
second_layernorm_files = sorted([file for file in files if 'second_layernorm' in file], key=lambda x: int(re.findall(r'\d+', x)[0]))

# 공통 설정
hist_ranges = (-15, 15)
device = torch.device('cpu')

def plot_histograms(layernorm_files, name):
    fig, axs = plt.subplots(3, 4, figsize=(24, 18), tight_layout=True)
    fig.suptitle(f'{name.capitalize()} Layernorm Histograms', fontsize=16)

    for i, layer_file in enumerate(layernorm_files):
        file_path = os.path.join(input_dir, layer_file)
        hidden_states = torch.load(file_path, map_location=device)
        tensor = hidden_states[0]
        data = tensor.numpy().flatten()

        row, col = divmod(i, 4)
        axs[row, col].hist(data, bins=100, alpha=0.75, range=hist_ranges)
        axs[row, col].set_title(f"{layer_file}")
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')

    plot_file_path = os.path.join(common_dir, f"./layernorm_histogram_and_topk_png/{name}_layernorm_histograms.png")
    plt.savefig(plot_file_path)
    plt.show()
    plt.close()

def plot_topk_values(layernorm_files, name, row_or_col):
    fig, axs = plt.subplots(3, 4, figsize=(24, 18), tight_layout=True)
    fig.suptitle(f'{name.capitalize()} Layernorm Top-10 Values', fontsize=16)

    for i, layer_file in enumerate(layernorm_files):
        file_path = os.path.join(input_dir, layer_file)
        hidden_states = torch.load(file_path, map_location=device)
        tensor = hidden_states[0]

        row, col = divmod(i, 4)

        #column 기준 top k
        if row_or_col == 'col' :
            topk_values = torch.topk(tensor, k=10, dim=0).values.numpy()

            for j in range(topk_values.shape[0]):  # 상위 값 개수 (10)
                y = topk_values[j, :]
                x = np.arange(tensor.size(1))  # column의 개수
                axs[row, col].scatter(x, y, marker='.')

        else :
            topk_values = torch.topk(first_tensor, k=10, dim=1).values.numpy()

            for j in range(topk_values.shape[1]):  # 상위 값 개수 (10)
                y = topk_values[:, j]
                x = np.arange(first_tensor.shape[0])  # row의 개수
                axs[row, col].scatter(x, y, marker='.')
        
        
        axs[row, col].set_title(f"{layer_file}")
        axs[row, col].set_xlabel('Column Index')
        axs[row, col].set_ylabel('Top-10 Values')

    plot_file_path = os.path.join(common_dir, f"./layernorm_histogram_and_topk_png/{name}_layernorm_topk.png")
    plt.savefig(plot_file_path)
    plt.show()
    plt.close()

# HISTOGRAM PLOT
plot_histograms(first_layernorm_files, 'first')
plot_histograms(second_layernorm_files, 'second')

# TOP 10 DATA PLOT
plot_topk_values(first_layernorm_files, 'first', 'col')
plot_topk_values(second_layernorm_files, 'second', 'col')
