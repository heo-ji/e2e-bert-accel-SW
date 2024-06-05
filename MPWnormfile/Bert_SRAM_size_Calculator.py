
def layernorm_sram(sequence_length,data):
    input_and_out = 768*data*sequence_length
    mean_invsqrt = 2*data*sequence_length
    weight = 768*data
    bias = 768*data
    sum = input_and_out+mean_invsqrt+weight+bias

    kb = sum/8/1024
    return kb

def ffn_sram(sequence_length,data):
    # systolic array 로 tiling해서 처리함
    input_and_f2out = 768*sequence_length*data
    bias1 = 768*data
    weight1 =  4*768*768*data
    f1_out = 4*768*data*sequence_length

    bias2 = 768*data
    weight2 =  768*4*768*data

    sum = input_and_f2out + bias1 + weight1 + f1_out + bias2 +weight2

    kb = sum/8/1024
    return kb
    



if __name__ == "__main__":

     #max_sequence length = 512token

    data = 16
    for n in range(1,17):
        sequence_length  = 32*n



        a=layernorm_sram(sequence_length,data)
        b = ffn_sram(sequence_length,data)/1024

        print("s_l =",sequence_length,"     layernorm:",a,"KB","   ffn:",b,"MB","    total :",(a/1024)+b,"MB")

   






