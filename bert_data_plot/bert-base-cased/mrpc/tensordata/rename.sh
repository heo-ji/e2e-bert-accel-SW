cd ..

for file in *; do
  if [ -f "$file" ]; then  # 디렉토리가 아닌 파일인지 확인
    filename=$(basename -- "$file")  # 파일의 기본 이름 추출
    extension="${filename##*.}"      # 확장자 추출
    name="${filename%.*}"            # 파일 이름에서 확장자 제외
    mv "$file" "./tensordata/13_${name}.$extension"  # 파일 이동 및 이름 변경]
  fi
done
