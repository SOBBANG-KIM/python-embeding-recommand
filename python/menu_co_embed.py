import json
import itertools
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# ✅ OpenSearch 연결
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

# ✅ 텍스트 임베딩 모델 (예: miniLM)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ 주문 로그 불러오기
orders = []
with open("../json/order_menu.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        orders.append(json.loads(line))

# ✅ 공동주문 카운트 계산
co_matrix = defaultdict(lambda: defaultdict(int))
menu_counts = defaultdict(int)

for order in orders:
    menus = order["menus"]
    for menu in menus:
        menu_counts[menu] += 1
    for a, b in itertools.combinations(sorted(menus), 2):
        co_matrix[a][b] += 1
        co_matrix[b][a] += 1

# ✅ OpenSearch 인덱스 생성
index_name = "menus"
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

index_body = {
    "settings": {
        "index": {
            "knn": True   # ✅ KNN 기능 활성화
        }
    },
    "mappings": {
        "properties": {
            "menu_name": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,  
                "method": {
                    "engine": "lucene",  # lucene / faiss / nmslib 중 선택
                    "name": "hnsw",
                    "space_type": "cosinesimil"
                }
            },
            "related_menus": {
                "type": "nested",
                "properties": {
                    "menu_name": {"type": "keyword"},
                    "co_occurrence_score": {"type": "float"}
                }
            }
        }
    }
}

client.indices.create(index=index_name, body=index_body)

# ✅ 메뉴별 데이터 업로드
all_menus = list(menu_counts.keys())

for menu in all_menus:
    embedding = model.encode(menu).tolist()
    related = []
    for other, count in co_matrix[menu].items():
        score = count / menu_counts[menu]  # 단순 확률 기반 점수
        related.append({"menu_name": other, "co_occurrence_score": score})

    doc = {
        "menu_name": menu,
        "embedding": embedding,
        "related_menus": related
    }

    client.index(index=index_name, body=doc)

print("✅ 메뉴 데이터 업로드 완료!")
