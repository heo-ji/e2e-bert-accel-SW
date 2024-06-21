'''
BATCH [0]

'''
import pdb
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cpu')
# 디렉토리 설정
common_dir = "/home/user/HJH/2024_git_nonlinear/vit_data_plot"
common_output_dir = os.path.join(common_dir, f"./attention_map_png")

# data
out_data_name = "patch16-224-cifar10/attention_out_data"
qk_data_name = "patch16-224-cifar10/attention_score_data"
softmax_data_name = "patch16-224-cifar10/attention_probs_data"

# FILE DIR
out_input_dir = os.path.join(common_dir, f"./tensor/{out_data_name}")
out_files = sorted(os.listdir(out_input_dir)) #torch.Size([4, 12, 197, 64])

qk_input_dir = os.path.join(common_dir, f"./tensor/{qk_data_name}")
qk_files = sorted(os.listdir(qk_input_dir)) #torch.Size([4, 12, 197, 197])

softmax_input_dir = os.path.join(common_dir, f"./tensor/{softmax_data_name}")
softmax_files = sorted(os.listdir(softmax_input_dir)) #torch.Size([4, 12, 197, 197])

qk_output_dir = os.path.join(common_output_dir, f"./attention_score")
softmax_output_dir = os.path.join(common_output_dir, f"./attention_prob")
out_output_dir = os.path.join(common_output_dir, f"./attention_out")


# file_path1 = os.path.join(out_input_dir, "layer0_attention_out.pt")
# file_path2= os.path.join(softmax_input_dir, "layer0_attention_probs.pt")
# file_path3 = os.path.join(qk_input_dir, "layer0_attention_score.pt")
# hidden_states1 = torch.load(file_path1, map_location=device)
# hidden_states2 = torch.load(file_path2, map_location=device)
# hidden_states3 = torch.load(file_path3, map_location=device)
# breakpoint()
# tensor = hidden_states1[0]


def plot_attention_matrices(files, input_dir, device, output_dir):
    for layer_file in files:
        file_path = os.path.join(input_dir, layer_file)
        hidden_states = torch.load(file_path, map_location=device)
        tensor = hidden_states[0]  # batch[0]의 데이터

        head_num, seq_len, seq_len_2 = tensor.shape  # (12, 197, 197) 형태

        fig, axs = plt.subplots(3, 4, figsize=(24, 18), tight_layout=True)
        fig.suptitle(f'{layer_file} Attention Head Matrices', fontsize=16)

        for head in range(head_num):
            matrix = tensor[head].detach().cpu().numpy()
            ax = axs[head // 4, head % 4]
            im = ax.imshow(matrix, cmap='Blues', aspect='auto')
            #im = ax.imshow(np.abs(matrix), cmap='Reds', aspect='auto')
            ax.set_title(f'Head {head}')
            fig.colorbar(im, ax=ax)

        plot_path = os.path.join(output_dir, f"{layer_file}_attention_matrices.png")
        plt.savefig(plot_path)
        plt.close()


#확인

plot_attention_matrices(qk_files, qk_input_dir, device, qk_output_dir)
plot_attention_matrices(softmax_files, softmax_input_dir, device, softmax_output_dir)


