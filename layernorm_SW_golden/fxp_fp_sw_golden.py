"""
: Qforamt 동작 = transformer src / custom_invsqrt.py / custom_norm.py 방식
input =   (total_bit = 16 , scale_factor(=frac_bit) = 8 ,  integer_bit = 8  )  
acc       (total_bit = 26 , scale_factor(=frac_bit) = 8 ,  integer_bit = 18 )
acc2      (total_bit = 34 , scale_factor(=frac_bit) = 8 ,  integer_bit = 26 )
mean      (total_bit = 16 , scale_factor(=frac_bit) = 8 ,  integer_bit = 8  )
mean2     (total_bit = 32 , scale_factor(=frac_bit) = 16 ,  integer_bit = 16 )
var       (total_bit = 24 , scale_factor(=frac_bit) = 16 , integer_bit = 8  )
invsqrt   (total_bit = 16 , scale_factor(=frac_bit) = 8 ,  integer_bit = 8  )

LUT s(8.8) * var(8.16) =>  + t(8.8)
LUT_mul =s(8.8) * var(8.16) => 16.24 -> 8.8

"""

import math
# =========================================================
# Qformat : floor + clip

def qformat_value(value, int_bits, frac_bits):
    scale = 2 ** frac_bits
    low = -(2 ** int_bits)
    high =  (2 ** int_bits) - 1.0 / scale
    v = math.floor(value * scale) / scale  # floor
    v = max(low, min(high, v))                # clip
    return v


def qfloor(value, frac_bits):
    """floor만"""
    scale = 2 ** frac_bits
    return math.floor(value * scale) / scale

# =========================================================


# =========================================================
# LUT  (원본 코드의 signed_CustomInvSqr 와 동일 로직,
#        단 d/s/t 값은 custom_invsqrt.py 방식으로 floor+clip)
# =========================================================
class LUT_InvSqrt:
    @staticmethod
    def hex_to_signed_float(hex_val, bits, frac_bits):
        """hex 비트패턴 → signed int → float"""
        signed_int = hex_val - (1 << bits) if hex_val & (1 << (bits-1)) else hex_val
        return signed_int / (2 ** frac_bits)

    def __init__(self):
        d_hex = [
            0x000487, 0x00048D, 0x0004A3, 0x0004C9, 0x0004CD, 0x0004FD, 0x00051F, 0x000555,
            0x0005AF, 0x00127D, 0x002113, 0x004049, 0x006A8F, 0x009B74, 0x00CE59, 0x01175C,
            0x018431, 0x023404, 0x03871C, 0x07CDCC, 0x07DED9, 0x1DC36A, 0x1FAD66, 0x222895,
        ]
        s_hex = [
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0xDBC7, 0xEF08, 0xF9F0, 0xFD73, 0xFEAD, 0xFF28, 0xFF73,
            0xFFA9, 0xFFCC, 0xFFE4, 0xFFF5, 0xFFC6, 0xFFFE, 0xFFEA, 0x0000,
        ]
        t_hex = [
            0x27F3, 0x2542, 0x21FD, 0x1E90, 0x1ABB, 0x1723, 0x1308, 0x0EDE,
            0x0ABB, 0x0645, 0x04E2, 0x0379, 0x0297, 0x0214, 0x01CA, 0x018D,
            0x0153, 0x011D, 0x00E8, 0x00AE, 0x021C, 0x005E, 0x02BC, 0x0000,
        ]

        # hex → signed float
        # d: 24bit, 8.16 해석
        self.d = [self.hex_to_signed_float(v, 24, 16) for v in d_hex]
        # s: 16bit, 8.8 해석
        self.s = [self.hex_to_signed_float(v, 16,  8) for v in s_hex]
        # t: 16bit, 8.8 해석
        self.t = [self.hex_to_signed_float(v, 16,  8) for v in t_hex]


    def forward(self, x):
        # d 와 x(variance값) 비교
        idx = len(self.d) - 1
        for i, d_val in enumerate(self.d):
            if x < d_val:
                idx = i
                break
        
        
        
        ## s*x = mul값을 (8.8)*(8.16)=>(16.24) 에서 8.8format 으로 정수부 saturation(overflow/underflow)!!
        
        s_mul_x = qformat_value(self.s[idx] * x, 8, 8)

        result  = qformat_value(s_mul_x + self.t[idx], 8, 8) ##clipping overflow 처리!!
        return result

# =========================================================
# SW FLOAT golden
# =========================================================
def sw_float_golden(values):
    """순수 float 연산"""
    n   = len(values)
    acc  = sum(values)
    acc2 = sum(v * v for v in values)
    mean  = acc  / n
    mean2 = acc2 / n
    var   = mean2 - mean * mean
    invsqrt = 1.0 / math.sqrt(var) if var > 0 else float('inf')

    normalized = [(v - mean) * invsqrt for v in values]

    return dict(acc=acc, acc2=acc2, mean=mean, mean2=mean2,
                var=var, invsqrt=invsqrt,normalized=normalized)


# =========================================================
# SW Qformat golden  (custom_norm.py forward_fxp88 동일 방식)
# =========================================================
def sw_qformat_golden(values, lut: LUT_InvSqrt):
    """
    floor + clip 기반 fixed-point 연산
    포맷: input 8.8 / acc 18.8 / acc2 26.8 / mean 8.8
          mean2 16.16 / var 8.16 / invsqrt 8.8
    """
    # --- input : 8.8 의 "768개"
    vals = [qformat_value(v, 8, 8) for v in values]
    n = len(vals)

    # --- acc : 18.8
    acc = sum(vals)
    acc = qformat_value(acc, 18, 8)

    # --- acc2 : 26.8  (x^2를 먼저 16.16에서 16.8로 rescale_소수부 버림_ 후 합산)
    acc2 = 0.0
    for v in vals:
        x2 = v * v                  # 16.16
        x2 = qformat_value(x2, 16, 8)    # → 16.8
        acc2 += x2
    acc2 = qformat_value(acc2, 26, 8)

    # --- mean : 8.8
    # d_model=768 , acc/d_model = acc / (2^8) * 0.33203125로 근사
    mean = acc / (2 ** 8)       #나눗셈연산 : 18.8->10.8 format 됨
    mean = qfloor(mean, 8)
    mean = mean * 0.33203125    #Q(10.8)*Q(0.8) = Q(10.16)

    mean = qformat_value(mean, 8, 8) # Q(10.16)-> 8.8 로 정수부saturation(overflow/underflow)!!

    # --- mean2 : 16.16
    mean2 = acc2 / (2 ** 8)            ##나눗셈연산 : 26.8->18.8 format 됨
    mean2 = qfloor(mean2, 8)
    mean2 = mean2 * 0.33203125      ##Q(18.8)*Q(0.8) = Q(18.16)

    mean2 = qformat_value(mean2, 16, 16)   # Q(18.16)-> Q(16.16) 로 정수부saturation(overflow/underflow)!!

    
    # --- E(x)^2 = 32bit(16.16)
    mean_sq = mean * mean
    mean_sq = qformat_value(mean_sq, 16, 16)

    # --- var : 8.16
    var = mean2 - mean_sq   
    var = qformat_value(var, 8, 16) # Q(16.16)->Q(8.16)로 정수부saturation(overflow/underflow)!!

    # --- eps (0.16)
    eps = 0.0000152587890625        # 1/2^16

    # --- invsqrt via LUT : 8.8
    v_eps = var + eps
    v_eps = qformat_value(v_eps, 8, 16)
    invsqrt = lut.forward(v_eps)

    # --- norm 8.8
    normalized = [(v - mean) * invsqrt for v in vals] # 8.8  * 8.8 = 16.16
    normalized = [qformat_value(norm, 8, 8) for norm in normalized] # Q(16.16)->Q(8.8)로 정수부saturation(overflow/underflow)!!


    return dict(acc=acc, acc2=acc2, mean=mean, mean2=mean2,
                var=var, invsqrt=invsqrt, normalized=normalized)
