#include <stdint.h>
#include <stdio.h>

float FastInvSqrt(float number) {
    int32_t i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = *(int32_t *) &y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - (i >> 1);                 // y0 of inverse square root
    y  = *(float *) &i;
    y  = y * (threehalfs - (x2 * y * y));       // 1st iteration of Newton-Raphson method

    return y;
}

int main(){
// int tokens = 128;
// int dmodel = 768;
// int dk = 64;
// int head = dmodel/dk;

    FILE *file;
    int16_t value;

    
    int count;
    int result;
    char ch;

    int32_t acc_sum=0;
    int32_t acc_var_sum=0;
    int16_t mean;
    int32_t var;
    float invsqrt;
    float out;

    // 파일 열기; "data.txt"는 실제 파일명으로 대체해야 함
    // file = fopen("11_FFN_layernorm_inputdata_int_16.txt", "r");
    // int dmodel =16;
    file = fopen("11_FFN_layernorm_inputdata_int_768.txt", "r");
    int dmodel =768;
    
    // 파일 열기에 실패한 경우
    if (file == NULL) {
        printf("파일을 열 수 없습니다.\n");
        
        return 1; // 비정상 종료
    }
    
    fscanf(file, "%c", &ch); // 첫 '[' 읽기

    

    while (fscanf(file, "%hd", &value) == 1) {
        //printf("int16 data = %hd\n", value);

        //stage1
        acc_sum += value;

        mean = acc_sum/dmodel;

        // 파일의 다음 문자가 ']'인지 확인
        long pos = ftell(file); // 현재 파일 위치 저장
        fscanf(file, "%c", &ch);
        if (ch == ']') {
            break; // ']'를 만나면 읽기 중단
        }
        fseek(file, pos, SEEK_SET); // ']'가 아니면 파일 포인터를 숫자 다음으로 되돌림
    }
    printf("acc_sum=%d, mean = %d\n",acc_sum, mean);


    fseek(file, 0, SEEK_SET);
    fscanf(file, "%c", &ch); // 첫 '[' 읽기
    while (fscanf(file, "%hd", &value) == 1) {

        //stage2
        acc_var_sum += (value-mean)*(value-mean);
        var = acc_var_sum/dmodel;

        // 파일의 다음 문자가 ']'인지 확인
        long pos = ftell(file); // 현재 파일 위치 저장
        fscanf(file, "%c", &ch);
        if (ch == ']') {
            break; // ']'를 만나면 읽기 중단
        }
        fseek(file, pos, SEEK_SET); // ']'가 아니면 파일 포인터를 숫자 다음으로 되돌림
    }
    printf("acc_var_sum=%d, var = %d\n",acc_var_sum, var);
    invsqrt = FastInvSqrt(var);
    printf("invsqrt=%f\n",invsqrt);

    fseek(file, 0, SEEK_SET);
    fscanf(file, "%c", &ch); // 첫 '[' 읽기
    while (fscanf(file, "%hd", &value) == 1) {
        // printf("int16 data - mean = %hd\n", value-mean);

        // //stage3
        // out = (value-mean) * invsqrt;
        // printf("normalize out = %f\n",out);

        // 파일의 다음 문자가 ']'인지 확인
        long pos = ftell(file); // 현재 파일 위치 저장
        fscanf(file, "%c", &ch);
        if (ch == ']') {
            break; // ']'를 만나면 읽기 중단
        }
        fseek(file, pos, SEEK_SET); // ']'가 아니면 파일 포인터를 숫자 다음으로 되돌림
    }

    // 파일 닫기
    fclose(file);
    return 0;

}