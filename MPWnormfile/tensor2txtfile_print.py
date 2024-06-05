import pdb
import numpy as np
import torch

def to_q8_8(values):
    scale_factor = 2**8  # 소수점을 8비트 옮기기 위한 스케일 팩터
    scaled_values = np.round(values * scale_factor)  # 가장 가까운 정수로 반올림
    
    # 8bit signed int 범위에 맞춰 클리핑
    q_formatted = np.clip(scaled_values, -2**15, 2**15 - 1).astype(np.int64)
    return q_formatted

def to_q18_8(values):
    scale_factor = 2**8  # 소수점을 8비트 옮기기 위한 스케일 팩터
    scaled_values = np.round(values * scale_factor)  # 가장 가까운 정수로 반올림
    
    # 8bit signed int 범위에 맞춰 클리핑
    q_formatted = np.clip(scaled_values, -2**25, 2**25 - 1).astype(np.int64)
    return q_formatted

def to_q26_8(values):
    scale_factor = 2**8  # 소수점을 8비트 옮기기 위한 스케일 팩터
    scaled_values = np.round(values * scale_factor)  # 가장 가까운 정수로 반올림
    
    # 26bit signed int 범위에 맞춰 클리핑
    q_formatted = np.clip(scaled_values, -2**33, 2**33 - 1).astype(np.int64)
    return q_formatted

def to_q16_16(values):
    scale_factor = 2**16 # 소수점을 10비트 옮기기 위한 스케일 팩터
    scaled_values = np.round(values * scale_factor)  # 가장 가까운 정수로 반올림
    
    # 8bit signed int 범위에 맞춰 클리핑
    q_formatted = np.clip(scaled_values, -2**31, (2**31) - 1).astype(np.int64)
    return q_formatted

# 파일로 저장
    np.savetxt(filename, numpy_array, fmt='%d', delimiter=',')
    print(f"Saved fixed point tensor to {filename}")


if __name__ == "__main__":

    device = torch.device('cpu')


    file_in =  "input_8_8.pt"  
    file1 = "acc_sum_18_8.pt"
    file2 = "acc_sum_x2_26_8.pt"     
        
    file3 =  "mean_8_8.pt"
    file4 =  "mean_x2_16_16.pt"
    file5 =  "var_8_16.pt"   
    file6 =  "invsqrt_8_8.pt"

    file7 =  "normalized_8_8.pt"

    file8 =  "bias_8_8.pt"
    file9 =  "biasedout_8_8.pt"

    file10 =  "weight_8_8.pt"
    file11 =  "weightedout_8_8.pt" 



    tensor_in = torch.load(file_in, map_location=device) #([8, 128, 768])
    acc_sum_18_8 = torch.load(file1, map_location=device) #([8, 128, 1])
    acc_sum_x2_26_8 = torch.load(file2, map_location=device) #([8, 128, 1])

    mean_8_8 = torch.load(file3, map_location=device) #([8, 128, 1])
    mean_x2_16_16 = torch.load(file4, map_location=device) #([8, 128, 1])
    var_16_16 = torch.load(file5, map_location=device) #([8, 128, 1])

    invsqrt_8_8 = torch.load(file6, map_location=device) #([8, 128, 1])

    normalized_8_8 = torch.load(file7, map_location=device) #([8, 128, 768])

    bias_8_8 = torch.load(file8, map_location=device)    #[768]
    biasedout_8_8 = torch.load(file9, map_location=device)  #([8, 128, 768])

    weight_8_8 = torch.load(file10, map_location=device)  #[768]
    weightedout_8_8 = torch.load(file11, map_location=device)  #([8, 128, 768])
    
 # see one batch
    tensor_in = np.array(tensor_in[0])                
    acc_sum_18_8 =   np.array(acc_sum_18_8[0])        
    acc_sum_x2_26_8 =   np.array(acc_sum_x2_26_8[0])  
    mean_8_8 =   np.array(mean_8_8[0])                
    mean_x2_16_16 =   np.array(mean_x2_16_16[0])      
    var_16_16 = np.array(var_16_16[0])                
    invsqrt_8_8 = np.array(invsqrt_8_8[0])            
    normalized_8_8 = np.array(normalized_8_8[0])      
    bias_8_8 = np.array(bias_8_8)                     
    biasedout_8_8 = np.array(biasedout_8_8[0])        
    weight_8_8 = np.array(weight_8_8)              
    weightedout_8_8 = np.array(weightedout_8_8[0])    

    input = to_q8_8(tensor_in) 
    acc = to_q18_8(acc_sum_18_8)
    acc2 = to_q26_8(acc_sum_x2_26_8)
    mean = to_q8_8(mean_8_8)
    mean2 = to_q16_16(mean_x2_16_16)
    var = to_q16_16(var_16_16)
    invsqrt = to_q8_8(invsqrt_8_8)

    normalized_8_8 = to_q8_8(normalized_8_8)
    bias_8_8    = to_q8_8(bias_8_8)
    biasedout_8_8 = to_q8_8(biasedout_8_8)
   
    weight_8_8  = to_q8_8(weight_8_8)
    weightedout_8_8  = to_q8_8(weightedout_8_8)

    #######
    # #token 128개 중 1개 -> 768개의 데이터
    #######
   
    # t1input = input[0]
    # t1acc = acc[0]
    # t1acc2 = acc2[0]
    # t1mean = mean[0]
    # t1mean2 = mean2[0]
    # t1var = var[0]
    # t1invsqrt = invsqrt[0]
    # t1normalized_8_8 = normalized_8_8[0]

    # t1bias_8_8 = bias_8_8
    # t1biasedout_8_8 = biasedout_8_8[0]
    # t1weight_8_8 = weight_8_8
    # t1weightedout_8_8 = weightedout_8_8[0]

    t1input = input[99]
    t1acc = acc[99]
    t1acc2 = acc2[99]
    t1mean = mean[99]
    t1mean2 = mean2[99]
    t1var = var[99]
    t1invsqrt = invsqrt[99]
    t1normalized_8_8 = normalized_8_8[99]

    t1bias_8_8 = bias_8_8
    t1biasedout_8_8 = biasedout_8_8[99]
    t1weight_8_8 = weight_8_8
    t1weightedout_8_8 = weightedout_8_8[99]

    with open('1row_normalized.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1normalized_8_8) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    with open('1row_input.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1input) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    with open('1row_bias.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1bias_8_8) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    with open('1row_bias_out.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1biasedout_8_8) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    with open('1row_weight.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1weight_8_8) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    with open('1row_weight_out.txt', 'w') as file:
        hex_1 = ''.join(f'{x & 0xFFFF:04X}\n' for x in t1weightedout_8_8) #8.8포멧의 16비트 값들
            
        file.write(hex_1)

    #breakpoint()

        # for i in range(0, len(t1input), 32):  # 32개씩 처리
        # # 한 줄에 들어갈 32개의 값 가져오기
        #     line_data = t1input[i:i+32]
        #     accum32fxp = np.sum(line_data)
        
        # # 각 값의 16진수 형식을 4자리로 맞추어 문자열로 변환
        #     hex_line = ''.join(f'{x & 0xFFFF:04X}' for x in line_data) #8.8포멧의 16비트 값들
            
        # # 파일에 쓰기
        #     file.write(hex_line + '\n')

    

    # breakpoint()
    with open('SW_one_row_output.txt', 'w') as file:
        hex = ''.join(f'acc(18.8) = {x & 0x3FF_FFFF:07X}' for x in t1acc) #18.8포멧의 26비트 값들
        file.write(hex + '\n')

    #breakpoint()
    #with open('acc2.txt', 'w') as file:
        hex = ' '.join(f'acc2(26.8) = {x & 0x3_FFFF_FFFF:010X}' for x in t1acc2) #26.8포멧의 34비트 값들
        file.write(hex + '\n')

        
    #with open('mean.txt', 'w') as file:
        hex = ' '.join(f'mean(8.8) = { x & 0xFFFF:04X}' for x in t1mean) #8.8포멧의 16비트 값들
        file.write(hex + '\n')

    #with open('mean2.txt', 'w') as file:
        hex = ' '.join(f'mean2(16.16) = {x & 0xFFFF_FFFF:08X}' for x in t1mean2) #16.16포멧의 비트 값들
        file.write(hex + '\n')

    #with open('var.txt', 'w') as file:
        hex = ' '.join(f'var(8.16) = {x & 0xFF_FFFF:06X}' for x in t1var) #8.16   16.16포멧의 26비트 값들
        file.write(hex + '\n')

    #with open('invsqrt.txt', 'w') as file:
        hex = ' '.join(f'invsqrt(8.8) = { x & 0xFFFF:04X}' for x in t1invsqrt) #16.10포멧의 26비트 값들
        file.write(hex + '\n')

        

    # breakpoint()
    


    # with open('square_sum.txt', 'w') as file:
    #     for i in range(0, len(t1input), 32):  # 32개씩 처리
    #     # 한 줄에 들어갈 32개의 값 가져오기
    #         line_data = t1input[i:i+32]
    #         #breakpoint()

            
            
    #         squared=((np.square(line_data)).astype(np.int64)) #처음32개
    #         squared_er8=((squared/256).astype(np.int64))
    #         squared_er8_sum=np.sum(squared_er8).astype(np.int64)  ###= 632448
    #         #squared_sum_er8=(np.sum(((np.square(line_data)).astype(np.int64)))/256).astype(np.int64)
            
    #         hex = ''.join(f'{squared_er8_sum & 0x3_FFFF_FFFF:010X}') #26.8포멧의 34비트 값들
    #         file.write(hex + '\n')

    # with open('sum.txt', 'w') as file:
    #     for i in range(0, len(t1input), 32):  # 32개씩 처리
    #     # 한 줄에 들어갈 32개의 값 가져오기
    #         line_data = t1input[i:i+32]
    #         accum32fxp = np.sum(line_data)
            
    #         hex_accum32fxp = ''.join(f'{accum32fxp & 0x3FF_FFFF:07X}') #8.8포멧의 16비트 값들
    #         file.write(hex_accum32fxp + '\n')



    #breakpoint()
            
           

    
