# co_menu.py

from opensearchpy import OpenSearch

# ✅ OpenSearch 클라이언트 정의
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),  # Docker 기본 계정 (변경 시 수정)
    use_ssl=False,
)

def update_related_menus(menu_name: str, related_list: list, index_name="menus"):
    resp = client.search(
        index=index_name,
        body={
            "query": {"term": {"menu_name": {"value": menu_name}}}
        }
    )

    if len(resp["hits"]["hits"]) == 0:
        print(f"❌ {menu_name} 문서를 찾을 수 없음")
        return

    doc_id = resp["hits"]["hits"][0]["_id"]

    client.update(
        index=index_name,
        id=doc_id,
        body={"doc": {"related_menus": related_list}}
    )

    print(f"✅ {menu_name} 관련메뉴 업데이트 완료")


# 🚀 실행 예시
if __name__ == "__main__":
    update_related_menus(
        "맛있는 불닭발",
        [
            {"menu_name": "소주", "co_occurrence_score": 0.7},
            {"menu_name": "계란찜", "co_occurrence_score": 0.4}
        ]
    )
