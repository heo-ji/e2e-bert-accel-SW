import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# 디렉토리 설정
input_dir = "/home/user/HJH/2024_nonlinear/transformers/src/transformers/models/vit/vit_input_range"
files = sorted(os.listdir(input_dir))


import pdb

'''
max heap = 부모가 자식보다 항상 크거나 같은 이진트리
'''
def heapify(arr, n, idx): 
    largest = idx   #가장 큰 인덱스를 idx로 초기화
    l = 2 * idx + 1 #왼쪽 자식 인덱스
    r = 2 * idx + 2 #오른쪽 자식 인덱스

    # 왼쪽 자식 노드가 현재 노드보다 크다면, largest를 왼쪽 자식 노드로 설정
    if l < n and arr[i] < arr[l]:
        largest = l
       
    # 오른쪽 자식 노드가 현재 largest 노드보다 크다면, largest를 오른쪽 자식 노드로 설정
    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # n//2  (n을 2로 나눈 몫) 부터 시작해서 0까지 하나씩 줄여감
    for idx in range(n // 2 - 1, -1, -1):
        heapify(arr, n, idx)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr

def top_k_heap_sort(input_tensor, k):
    input_array = input_tensor.numpy()
    flattened_array = input_array.flatten()
    sorted_array = heap_sort(flattened_array)
    top_k_elements = sorted_array[-k:]
    return torch.tensor(top_k_elements)

'''
quick search
'''
def partition(arr, low, high):
    i = (low - 1)  # 작은 요소의 인덱스
    pivot = arr[high]  # 피벗 요소를 선택

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]  # swap

    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # 피벗을 올바른 위치로 이동
    return (i + 1)

def quick_sort(arr, low, high):
    if len(arr) == 1:
        return arr
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)
    return arr

def top_k_quick_sort(input_tensor, k):
    input_array = input_tensor.numpy()  # 텐서를 numpy 배열로 변환
    flattened_array = input_array.flatten()  # 1차원 배열로 플래튼
    n = len(flattened_array)
    sorted_array = quick_sort(flattened_array, 0, n - 1)  # 퀵 정렬
    top_k_elements = sorted_array[-k:]  # 상위 k 요소 추출
    return torch.tensor(top_k_elements)  # 텐서로 변환하여 반환


'''
spatten quick select -> top k
fifo depth =64

'''

def quick_select_top_k(input_tensor, k):
    input_array = input_tensor.numpy().flatten()
    target = k
    num_eq_pivot = 0
    # breakpoint()
    # Initialize FIFO_L with input_array, FIFO_R is empty
    FIFO = list(input_array)
    FIFO_L = list(input_array)
    FIFO_R = []
    
    while True:
        if len(FIFO_R) + num_eq_pivot <= target:
            target = target - len(FIFO_R) - num_eq_pivot
            num_eq_pivot = len(FIFO_R)
            FIFO_R = []
            select = 0
            pivot = FIFO_L[np.random.randint(len(FIFO_L))]
            state_run(FIFO_L, FIFO_R, pivot, num_eq_pivot)
            continue
        elif len(FIFO_R) > target:
            FIFO_L = []
            select = 1
            pivot = FIFO_R[np.random.randint(len(FIFO_R))]
            state_run(FIFO_L, FIFO_R, pivot, num_eq_pivot)
            continue
        else:
            kth_largest = pivot
            num_eq_kth_largest = target - len(FIFO_R)
            return torch.tensor([kth_largest] * num_eq_kth_largest)
    
    return torch.tensor(top_k_elements)



def state_run(FIFO_L, FIFO_R, pivot, num_eq_pivot):
    for item in FIFO_L:
        if item < pivot:
            FIFO_L.append(item)
        elif item > pivot:
            FIFO_R.append(item)
        else:
            num_eq_pivot += 1

# Example usage
input_tensor = torch.randn(1, 20)
k = 4

top_k_elements_quick_select = quick_select_top_k(input_tensor, k)
print("Top-k elements using Quick Select:", top_k_elements_quick_select)




# # Example usage
# input_tensor = torch.randn(129, 768)
# k = 10
# top_k_elements_heap = top_k_heap_sort(input_tensor, k)
# print("Top-k elements using Heap Sort:", top_k_elements_heap)


# D_s = 384  # example value for the number of significant dimensions
# N_t = 5  # sampling rate for trivial dimensions

# # 각 파일을 읽고 boxplot 생성
# for file in files:
#     if file.endswith(".pt"):
#         file_path = os.path.join(input_dir, file)
#         hidden_states = torch.load(file_path)
#         input = hidden_states[0].cpu().np()


        
#         # scale_factor_8 = 2**8
#         # scale_factor_10 = 2**10
#         # scale_factor_16 = 2**16

#         # #input = 8.8
#         # input = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
#         # input = torch.clip(input, -2**7, 127.99609375) #정수부8.8
#         # #torch.save(input,'/home/user/HJH/transformers/src/MPWnormfile/input_8_8.pt') ##저장

#         # Algorithm implementation
#         B, N, D = input.shape
#         device = input.device
#         D_t = D - self.D_s
#         Y = torch.zeros_like(input)

#         for b in range(B):
#             for n in range(N):
#                 X = input[b, n, :]

#                 # Identify significant dimensions
#                 abs_X = torch.abs(X)
#                 sig_values, indices = torch.topk(abs_X, self.D_s)
#                 sig_mask = torch.zeros_like(X, dtype=torch.bool)
#                 sig_mask[indices] = True

#                 # Path 1: Significant dimensions
#                 sumX_sig = torch.sum(X[sig_mask])
#                 sumX2_sig = torch.sum(X[sig_mask] ** 2)

#                 # Path 2: Trivial dimensions
#                 tri_mask = ~sig_mask
#                 sumX_tri = torch.sum(X[tri_mask])

#                 # Sample every Nt-th element for trivial dimensions' square sum
#                 sampled_tri_indices = torch.nonzero(tri_mask).squeeze(1)[::self.N_t]
#                 sumX2_tri = torch.sum(X[sampled_tri_indices] ** 2) * D_t

#                 # Calculate mean and variance
#                 meanX = (sumX_sig + sumX_tri) / D
#                 varX = (sumX2_sig + sumX2_tri) / D - meanX ** 2

#                 # Normalization
#                 Y[b, n, :] = (X - meanX) / torch.sqrt(varX + self.eps)

#         normalized = Y