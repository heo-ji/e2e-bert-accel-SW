
import subprocess
import os

# 원본 스크립트 plt , replace("10_"
# original_script = './plot_a_laynorm_plt.py'
# original_script = './plot_FFN_laynorm_plt.py'
# original_script = './plot_FFN_plt.py'

original_script = './data_analyze.py'

# 1부터 12까지 반복
for i in range(1, 13):
    # 원본 스크립트 파일 읽기
    with open(original_script, 'r') as file:
        original_content = file.read()
    
    # "1_" 문자열을 i에 해당하는 숫자로 변경
    modified_content = original_content.replace("11_", f"{i}_")
    
    # 변경된 내용을 임시 스크립트 파일로 저장
    modified_script = f'temp_script_{i}.py'
    with open(modified_script, 'w') as file:
        file.write(modified_content)
    
    # 변경된 스크립트 실행
    subprocess.run(['python', modified_script], check=True)
    
    # 임시 스크립트 파일 삭제 (선택 사항)
    os.remove(modified_script)  # 필요한 경우 주석 해제
