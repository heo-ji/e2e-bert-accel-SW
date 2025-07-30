'''
value_name(Qformat) = example value
--------------------------------
accumulate value(18.8) = 3FFCF4E
accumulate squared value (26.8) = 000004EC45

mean(8.8) = FFEF
mean_by squared value (16.16) = 0001A25C

variance(8.16) = 01A13B
inverse square root of variance (8.8) = 00C7

'''


'''
### generate random value


import random
import statistics

def generate_random_values_with_variance(num_values, variance_limit_min, variance_limit_max):
    while True:
        # Generate random values within a typical range, for example, -128 to 128
        random_values = [random.uniform(-128, 128) for _ in range(num_values)]

        # Calculate the variance of the generated values
        variance = statistics.pvariance(random_values)

        # Check if the variance is within the specified limits
        if variance_limit_min <= variance <= variance_limit_max:
            return random_values

# 8.16포맷의 variance로 인한 limit
variance_limit_min = 0
variance_limit_max = 127.99998474121094

#LUT로 인한 variance limit , inverse square root 값이 0000이 안나오는 값확인용
variance_limit_min = 31
variance_limit_max = 31.677


# Get the number of desired values from the user
num_values = int(input("The number of desired values, Please enter: "))

# Generate random values with the desired variance
random_values = generate_random_values_with_variance(num_values, variance_limit_min, variance_limit_max)

# Display the generated random values and their variance
print(random_values)

print("\nVariance of generated values:", statistics.pvariance(random_values))


def float_rounding_to_fixedformat(values,scale_factor):
  result = []
  for value in values:
    round_value=round(value * (2 ** scale_factor))/(2 ** scale_factor)
    result.append(round_value)
  return result


values = float_rounding_to_fixedformat(random_values,8)
print("\nvalues = ",values)
'''

'''
######SW floating value check

def float_to_fixed_hex(value, total_bits, scale_factor):
    # Apply scaling factor with rounding
    fixed_point_value = round(value * (2 ** scale_factor))

    # If the value is negative, convert it to two's complement form
    if fixed_point_value < 0:
        fixed_point_value = (1 << total_bits) + fixed_point_value

    # Apply mask to ensure the value fits within the specified bit size
    mask = (1 << total_bits) - 1
    fixed_point_value &= mask

    # Convert to a fixed-length hexadecimal string
    return format(fixed_point_value, '0' + str(total_bits // 4) + 'X')

def cal_float(float_values):
  acc = 0
  acc2 = 0

  for value in float_values:
    acc += value
    acc2 += value * value

  mean = acc / len(float_values)
  mean2 = acc2 / len(float_values)
  var = mean2 - mean * mean
  inv_sqrt_var = 1 / (var ** 0.5)

  return acc, acc2, mean, mean2, var, inv_sqrt_var

print(values)
acc , acc2 , mean , mean2 , var , inv_sqrt_var = cal_float(values)

print("SW output")
print("acc       (total = 26 , scale_factor = 8)  = ", float_to_fixed_hex(acc, 26, 8))
print("acc2      (total = 34 , scale_factor = 8)  = ", float_to_fixed_hex(acc2, 34, 8))
print("mean      (total = 16 , scale_factor = 8)  = ", float_to_fixed_hex(mean, 16, 8))
print("mean2     (total = 32 , scale_factor = 16)  = ", float_to_fixed_hex(mean2, 32, 16))
print("var       (total = 24 , scale_factor = 16) = ", float_to_fixed_hex(var, 24, 16))
print("invsqrt   (total = 16 , scale_factor = 8)  = ", float_to_fixed_hex(inv_sqrt_var, 16, 8))

print("\n SW output floating point ")
print("acc       = ", acc)
print("acc2      = ", acc2)
print("mean      = ", mean)
print("mean2     = ", mean2)
print("var       = ", var)
print("invsqrt   = ", inv_sqrt_var)
'''

"""
### hardware value check 
class signed_CustomInvSqr:
    def __init__(self):
        # Initialize d, s, t with the provided signed hexadecimal values
        self.d = [
            0x000487, 0x00048D, 0x0004A3, 0x0004C9, 0x0004CD, 0x0004FD, 0x00051F, 0x000555,
            0x0005AF, 0x00127D, 0x002113, 0x004049, 0x006A8F, 0x009B74, 0x00CE59, 0x01175C,
            0x018431, 0x023404, 0x03871C, 0x07CDCC, 0x07DED9, 0x1DC36A, 0x1FAD66, 0x222895
        ]

        self.s = [
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0xDBC7, 0xEF08, 0xF9F0, 0xFD73, 0xFEAD, 0xFF28, 0xFF73,
            0xFFA9, 0xFFCC, 0xFFE4, 0xFFF5, 0xFFC6, 0xFFFE, 0xFFEA, 0x0000
        ]

        self.t = [
            0x27F3, 0x2542, 0x21FD, 0x1E90, 0x1ABB, 0x1723, 0x1308, 0x0EDE,
            0x0ABB, 0x0645, 0x04E2, 0x0379, 0x0297, 0x0214, 0x01CA, 0x018D,
            0x0153, 0x011D, 0x00E8, 0x00AE, 0x021C, 0x005E, 0x02BC, 0x0000
        ]

    def to_signed(self, value, bits):
        #Convert an unsigned integer to a signed integer using two's complement.
        if value & (1 << (bits - 1)):
            return value - (1 << bits)
        return value

    def to_unsigned(self, value, bits):
        #Convert a signed integer to an unsigned integer.
        return value & ((1 << bits) - 1)

    def search_sorted(self, d, x):
        # x is still a hexadecimal string, so we need to convert d elements to hex and compare
        for i, val in enumerate(d):
            if int(x, 16) < val:  # Convert x to an integer for comparison
                return i
        return len(d) - 1

    def forward(self, x_hex):
        idx = self.search_sorted(self.d, x_hex)

        # Convert x_hex to an integer for multiplication
        x = int(x_hex, 16)

        # Convert self.s[idx] to signed before multiplication
        s_signed = self.to_signed(self.s[idx], 16)
        x_signed = self.to_signed(x, 24)
        # Perform signed multiplication
        s_mult_x = s_signed * x_signed

        # print(f"x (hex): {x_hex}")
        # print(f"s[idx] (signed): {s_signed}")
        # print(f"s_mult_x (hex, signed): {format(s_mult_x, '016X')}")

        # s_mult_x is now in 16.24 format
        # Extract the middle 16 bits to get the 8.8 result
        result_8_8 = (s_mult_x >> 16) & 0xFFFF
        result_8_8_signed = self.to_signed(result_8_8, 16)
        #print(f"result_8_8 (hex): {format(result_8_8, '04X')}")

        # Convert self.t[idx] to signed
        t_signed = self.to_signed(self.t[idx], 16)
        #print(f"t[idx] (signed): {t_signed}")

        # Perform signed addition
        signed_result = result_8_8_signed + t_signed
        #print(f"signed result (before converting to unsigned): {signed_result}")

        # Convert the result back to unsigned for the final output
        result = self.to_unsigned(signed_result, 16)
        #print(f"final result (hex): {format(result, '04X')}")
        return result

    def calculate(self, hex_val):
        # hex_val is passed directly to forward without converting to an integer
        result = self.forward(hex_val)
        return format(result & 0xFFFF, '04X')  # Return the result as a 16-bit hex


# inv_sqrt_calc = signed_CustomInvSqr()
# re=inv_sqrt_calc.calculate("1f9fff")
# print(re)


def float_to_fixed(value, scale_factor):
    return round(value * (2 ** scale_factor))

def fixed_to_float(fixed_value, scale_factor):
    return fixed_value / (2 ** scale_factor)

def fixed_multiply(fixed1, fixed2, scale_factor):
    return (fixed1 * fixed2) >> scale_factor

# def fixed_to_hex(fixed_value, total_bits):
#     return format(fixed_value, '0' + str(total_bits // 4) + 'X')
def fixed_to_hex(fixed_value, total_bits):
    # Calculate the number of bits available
    if fixed_value < 0:
        # If the value is negative, convert it to its two's complement form
        fixed_value = (1 << total_bits) + fixed_value

    # Format the value as a hexadecimal string with the correct number of digits
    return format(fixed_value & ((1 << total_bits) - 1), '0' + str(total_bits // 4) + 'X')

def cal(values):
    # Define scale factors for each value
    scale_factor_acc = 8
    scale_factor_acc2 = 8
    scale_factor_mean = 8
    scale_factor_mean2 = 16
    scale_factor_var = 16
    scale_factor_invsqrt = 8

    # Convert input values to fixed-point using the mean's scale factor
    fixed_values = [float_to_fixed(val, scale_factor_mean) for val in values]

    # Calculate acc (sum of values) using acc's scale factor
    acc = sum(float_to_fixed(val, scale_factor_acc) for val in values)

    # Calculate acc2 (sum of squares) using acc2's scale factor
    acc2 = sum(fixed_multiply(float_to_fixed(val, scale_factor_acc2), float_to_fixed(val, scale_factor_acc2), scale_factor_acc2) for val in values)

    # Calculate mean (acc / len(values)) using mean's scale factor
    mean = acc // len(fixed_values)


    # Calculate mean2 (acc2 / len(values)) using mean2's scale factor
    mean2 = acc2 * (2 ** 8) // len(fixed_values)


    # Calculate variance using var's scale factor
    var = (mean2 - mean* mean)
    ## fixed_to_hex(var, 24))여기에서 위 8비트 어처피 안보기 때문에, var 계산하고 shift 안해도 되는거

    # Calculate inverse square root using CustomInvSqrt class
    inv_sqrt_calc = signed_CustomInvSqr()

    # Convert variance to hex and calculate invsqrt using var's scale factor
    var_hex = fixed_to_hex(var, 24)

    invsqrt_hex = inv_sqrt_calc.calculate(var_hex)

    return acc, acc2, mean, mean2, var, invsqrt_hex



acc, acc2, mean, mean2, var, invsqrt_hex = cal(values)

# Print results in hexadecimal format
print("HW output")
print("acc       (total = 26 , scale_factor = 8):", fixed_to_hex(acc, 26))
print("acc2      (total = 34 , scale_factor = 8):", fixed_to_hex(acc2, 34))
print("mean      (total = 16 , scale_factor = 8):", fixed_to_hex(mean, 16))
print("mean2     (total = 32 , scale_factor = 16):", fixed_to_hex(mean2, 32))
print("var       (total = 24 , scale_factor = 16):", fixed_to_hex(var, 24)) #
print("invsqrt   (total = 16 , scale_factor = 8):", invsqrt_hex)

"""