# 수정된 코드 부분

import struct
import csv
import numpy as np

def float_to_binary_with_details(number):
    """
    주어진 FP32 실수를 IEEE 754 표준에 따른 32비트 이진 문자열로 변환하며,
    sign, exponent, mantissa를 "_"로 구분하여 표현합니다.

    Args:
    number (float): 변환할 FP32 실수.

    Returns:
    str: 변환된 이진 문자열, sign_exponent_mantissa 형태.
    """
    # 실수를 4바이트의 이진 데이터로 패킹합니다.
    packed = struct.pack('f', number)
    # 패킹된 이진 데이터를 정수로 변환합니다.
    packed_int = int.from_bytes(packed, 'little', signed=False)
    # 정수를 이진 문자열로 변환합니다. 앞의 '0b'를 제거하고, 32비트가 되도록 앞을 0으로 채웁니다.
    binary_string = f"{packed_int:032b}"
    
    # sign (1 bit), exponent (8 bits), mantissa (23 bits)로 구분합니다.
    sign = binary_string[0]
    exponent = binary_string[1:9]
    mantissa = binary_string[9:]
    
    # 구분자 "_"를 사용하여 이진 문자열을 합칩니다.
    detailed_binary_string = f"{sign}_{exponent}_{mantissa}"
    
    return detailed_binary_string

def float_to_binary_fixed_8_8(number):
    scale_factor = 2**8

    # 무한대 값인 경우 "inf"로 처리
    if number == float('inf'):
        return "inf"
    elif number == float('-inf'):
        return "-inf"

    try:
        scaled_values = np.round(number * scale_factor)
        q_formatted = np.clip(scaled_values, -2**15, (2**15) - 1).astype(np.int64)
        int16_val = int(q_formatted) & 0xFFFF  # 16비트 범위로 제한
        # scaled_number = number * (2**8)
        # # 곱한 결과를 int16로 변환합니다.
        # int16_val = int(scaled_number) & 0xFFFF  # 16비트 범위로 제한
    except OverflowError:
        # 변환 과정에서 오버플로가 발생하는 경우 "inf"로 처리
        return "inf"

    # int16 값을 이진 문자열로 변환합니다.
    binary_string = f"{int16_val:04X}"
    
    # 상위 8비트를 정수 부분, 하위 8비트를 소수 부분으로 구분합니다.
    integer_part = binary_string[:2]
    fractional_part = binary_string[2:]
    
    # 정수 부분과 소수 부분 사이에 "_"를 넣어 구분합니다.
    detailed_binary_string = f"16'sh{integer_part}_{fractional_part}"
    
    return detailed_binary_string

def float_to_binary_fixed_8_16(number):
    scale_factor16 = 2**16
    # 무한대 값인 경우 "inf"로 처리
    if number == float('inf'):
        return "inf"
    elif number == float('-inf'):
        return "-inf"

    try:
        scaled_values = np.round(number * scale_factor16)
        q_formatted = np.clip(scaled_values, -2**31, (2**31) - 1).astype(np.int64)
        int24_val = int(q_formatted) & 0xFF_FFFF  # 32비트 범위로 제한
        # scaled_number = number * (2**16)
        # # 곱한 결과를 int16로 변환합니다.
        # int32_val = int(scaled_number) & 0xFF_FFFF  # 24비트 범위로 제한
    except OverflowError:
        # 변환 과정에서 오버플로가 발생하는 경우 "inf"로 처리
        return "inf"

    # int16 값을 이진 문자열로 변환합니다.
    binary_string = f"{int24_val:06X}"
    
    #상위 8트를 정수 부분, 하위 비트를 소수 부분으로 구분합니다.
    integer_part = binary_string[:2]
    fractional_part = binary_string[2:]
    
    # 정수 부분과 소수 부분 사이에 "_"를 넣어 구분합니다.
    detailed_binary_string = f"{integer_part}_{fractional_part}"
    
    return detailed_binary_string

def float_to_binary_fixed_9_23(number):
    # 무한대 값인 경우 "inf"로 처리
    if number == float('inf'):
        return "inf"
    elif number == float('-inf'):
        return "-inf"

    try:
        scaled_number = number * (2**23)
        # 곱한 결과를 int32로 변환합니다.
        int32_val = int(scaled_number) & 0xFFFFFFFF  # 32비트 범위로 제한
    except OverflowError:
        # 변환 과정에서 오버플로가 발생하는 경우 "inf"로 처리
        return "inf"

    # int32 값을 이진 문자열로 변환합니다.
    binary_string = f"{int32_val:032b}"
    
    # 상위 9비트를 정수 부분, 하위 23비트를 소수 부분으로 구분합니다.
    integer_part = binary_string[:9]
    fractional_part = binary_string[9:]
    
    # 정수 부분과 소수 부분 사이에 "_"를 넣어 구분합니다.
    detailed_binary_string = f"{integer_part}_{fractional_part}"
    
    return detailed_binary_string

# CSV 파일 읽기 및 변환 함수 수정
def read_and_convert_csv(input_filename, output_filename, signal):
    with open(input_filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)

    converted_lines = []
    for row in data:
        if signal == 0:
            # signal이 0일 때는 float_to_binary_with_details 함수를 적용합니다.
            converted_row = [float_to_binary_with_details(float(value)) for value in row]
        elif signal == 88:
            # signal이 88일 때는 float_to_binary_fixed_8_8 함수를 적용합니다.
            converted_row = [float_to_binary_fixed_8_8(float(value)) for value in row]
        elif signal == 816:
            # signal이 1일 때는 float_to_binary_fixed_9_23 함수를 적용합니다.
            converted_row = [float_to_binary_fixed_8_16(float(value)) for value in row]
        elif signal == 3:
            # signal이 1일 때는 float_to_binary_fixed_9_23 함수를 적용합니다.
            converted_row = [float_to_binary_fixed_9_23(float(value)) for value in row]
        else:
            raise ValueError("Signal must be 0 ~ 3.")
        converted_lines.append(converted_row)

    # 변환된 데이터를 새 CSV 파일에 쓰기
    with open(output_filename, 'w', newline='') as d:
        writer = csv.writer(d)
        writer.writerows(converted_lines)
        
        

# 함수 호출 예시 (실제 실행을 위해서는 주석을 해제하세요)
read_and_convert_csv("/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_d.csv", "/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/32inv_SQRT_d_fixed.csv", 816)
read_and_convert_csv("/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_s.csv", "/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/32inv_SQRT_s_fixed.csv", 88)
read_and_convert_csv("/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_t.csv", "/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/32inv_SQRT_t_fixed.csv", 88)
