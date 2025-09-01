from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# ✅ OpenSearch 연결
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

# ✅ 임베딩 모델
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ 신메뉴 등록 함수
def add_new_menu(menu_name: str, index_name="menus"):
    embedding = model.encode(menu_name).tolist()

    doc = {
        "menu_name": menu_name,
        "embedding": embedding,
        "related_menus": []  # 신메뉴는 공동주문 데이터가 없음
    }

    resp = client.index(index=index_name, body=doc)
    print(f"✅ 신메뉴 등록 완료: {menu_name}, document_id={resp['_id']}")

# 🚀 사용 예시
add_new_menu("맛있는 불닭발")
