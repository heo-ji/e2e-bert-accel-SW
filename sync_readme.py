import os
from notion_client import Client

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
BLOCK_ID = os.environ.get("NOTION_BLOCK_ID") # README가 들어갈 위치

notion = Client(auth=NOTION_TOKEN)

def sync():
    # 1. README 파일 읽기
    with open("README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()

    # 2. 지정한 블록 내부의 기존 내용 삭제
    existing_content = notion.blocks.children.list(block_id=BLOCK_ID)
    for block in existing_content["results"]:
        notion.blocks.delete(block_id=block["id"])

    # 3. README 내용을 2000자 단위로 쪼개기 (노션 API 글자수 제한)
    chunks = [readme_text[i:i+2000] for i in range(0, len(readme_text), 2000)]
    
    new_blocks = [
        {
            "object": "block",
            "type": "code", # 마크다운 형식을 유지하려면 code 블록이 가장 깔끔합니다.
            "code": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}],
                "language": "markdown"
            }
        } for chunk in chunks
    ]

    # 4. 지정한 블록 안에 새 내용 추가
    notion.blocks.children.append(block_id=BLOCK_ID, children=new_blocks)
    print("특정 블록에 README 동기화 완료!")

if __name__ == "__main__":
    sync()
