import scipy.stats
import numpy as np

def to_q8_8(values):
    scale_factor = 2**8  # 소수점을 8비트 옮기기 위한 스케일 팩터
    scaled_values = np.round(values * scale_factor)  # 가장 가까운 정수로 반올림
    
    # 8bit signed int 범위에 맞춰 클리핑
    q_formatted = np.clip(scaled_values, -2**15, 2**15 - 1).astype(np.int64)
    return q_formatted

if __name__ == "__main__":
    lower = -127
    upper = 127.9
    mu = 4
    sigma = 2 #var=4 , invsqrt=0.5
    N = 768

    samples = scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)#.toarray()

    row1 = to_q8_8(samples)

    with open('32Xinput.txt', 'a') as file:
        for x in row1 :
            hex_1 = ''.join(f'{x & 0xFFFF:04X}' for i in range(32)) #8.8포멧의 16비트 값들
            file.write(hex_1 + '\n')