"""
Qformat(SW FLOAT golden hex)  과  SW Qformat golden hex  인지 확인?

SW FLOAT golden vs SW Qformat golden 비교
- var를 0.5~40.0까지 sweep
- float golden을 fxp로 양자화한 값 vs qformat golden 값 비교
- 어느 필드에서 차이가 나는지 확인

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

import random
import statistics
import math
import fxp_fp_sw_golden as SW_golden
#
#return dict(acc=acc, acc2=acc2, mean=mean, mean2=mean2,
#                var=var, invsqrt=invsqrt,normalized=normalized)
#

# =========================================================
# hex 변환
# =========================================================
FIELDS = [
    ('acc',        18,  8),
    ('acc2',       26,  8),
    ('mean',        8,  8),
    ('mean2',      16, 16),
    ('var',         8, 16),
    ('invsqrt',     8,  8),
]

def to_hex(value, int_bits, frac_bits):
    total_bits = int_bits + frac_bits

    make_int_value = value * (2 ** frac_bits)
    fixed = math.floor(make_int_value) #

    if fixed < 0:
        fixed = (1 << total_bits) + fixed #2의보수 음수 , 총비트는 (1 + total_bits)

    fixed &= (1 << total_bits) - 1 #(1 << total_bits)=> (1000...0) - 1 = (0111...1)
    return format(fixed, f'0{total_bits // 4}X') #4비트 = 16진수 한자리


# =========================================================
# 비교 : float golden을 fxp로 양자화한 hex vs qformat golden hex
# =========================================================
def compare_golden(float_res, qfmt_res):
    diff_fields = []
    for name, ib, fb in FIELDS:
        fv = float_res[name]
        qv = qfmt_res[name]
        if not math.isfinite(fv):
            diff_fields.append(name)
            continue
        # float golden도 동일 fxp포맷으로 양자화 후 hex 비교
        fh = to_hex(SW_golden.qformat_value(fv, ib, fb), ib, fb)
        qh = to_hex(qv, ib, fb)
        if fh != qh:
            diff_fields.append(name)
    return diff_fields


# =========================================================
# target_var에 근접한 랜덤값 생성
# =========================================================
def generate_values_for_target_var(num_values, target_var, tolerance=0.5, max_tries=500000):
    for _ in range(max_tries):
        vals = [random.uniform(-128, 128) for _ in range(num_values)]
        var  = statistics.pvariance(vals)
        if abs(var - target_var) <= tolerance:
            return vals
    raise RuntimeError(f"target_var={target_var} 생성 실패")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    NUM_VALUES = 768
    TRIALS     = 3   # 각 var마다 몇 번 시도
    lut        = SW_golden.LUT_InvSqrt()

    var_list = [round(v * 0.5, 1) for v in range(1, 81)]  # 0.5 ~ 40.0

    print("=" * 70)
    print(f"  SW FLOAT golden  vs  SW Qformat golden  (N={NUM_VALUES})")
    print(f"  var sweep: 0.5 ~ 40.0, step=0.5, trials={TRIALS}")
    print("=" * 70)
    print(f"  {'target_var':>10s}  {'actual_var':>10s}  result")
    print("  " + "-" * 60)

    for target_var in var_list:
        results = []
        for _ in range(TRIALS):
            try:
                vals = generate_values_for_target_var(NUM_VALUES, target_var)
            except RuntimeError:
                results.append("생성실패")
                continue

            actual_var  = statistics.pvariance(vals)
            float_res   = SW_golden.sw_float_golden(vals)
            qfmt_res    = SW_golden.sw_qformat_golden(vals, lut)
            diff_fields = compare_golden(float_res, qfmt_res)

            if diff_fields:
                results.append(f"diff {diff_fields}")
            else:
                results.append("correct")

        # TRIALS 중 하나라도 diff 있으면 출력
        any_diff = any("diff" in r for r in results)
        if any_diff:
            for i, r in enumerate(results):
                print(f"  {target_var:>10.1f}  trial {i+1}: {r}")
        else:
            print(f"  {target_var:>10.1f}  {'':>10s}  correct")