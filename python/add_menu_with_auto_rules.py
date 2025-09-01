import json
import itertools
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "menus"
ORDERS_FILE = "../json/order_menu.jsonl"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_cooccurrence_matrix(orders):
    co_matrix = defaultdict(lambda: defaultdict(int))
    menu_counts = defaultdict(int)

    for order in orders:
        menus = order["menus"]
        for menu in menus:
            menu_counts[menu] += 1
        for a, b in itertools.combinations(sorted(menus), 2):
            co_matrix[a][b] += 1
            co_matrix[b][a] += 1

    return co_matrix, menu_counts


def add_new_menu_with_auto_rules(menu_name: str, index_name=INDEX_NAME):
    # ✅ 1. 주문 로그 읽기
    orders = []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            orders.append(json.loads(line))

    co_matrix, menu_counts = build_cooccurrence_matrix(orders)

    # ✅ 2. 신메뉴 임베딩
    embedding = model.encode(menu_name).tolist()

    # ✅ 3. 공동주문 데이터 있으면 직접 반영
    related_list = []
    if menu_name in co_matrix:
        for other, count in co_matrix[menu_name].items():
            score = count / menu_counts[menu_name]
            related_list.append({
                "menu_name": other,
                "co_occurrence_score": round(score, 3)
            })

    # ✅ 4. 공동주문 데이터가 없으면 → 임베딩 기반 최근접 메뉴 룰 복사
    if not related_list:
        resp = client.search(
            index=index_name,
            body={
                "size": 1,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": 1
                        }
                    }
                }
            }
        )
        if resp["hits"]["hits"]:
            nearest_menu = resp["hits"]["hits"][0]["_source"]
            related_list = nearest_menu.get("related_menus", [])
            print(f"📌 공동주문 데이터 없음 → 최근접 메뉴({nearest_menu['menu_name']}) 룰 복사")

    # ✅ 5. OpenSearch에 저장
    doc = {
        "menu_name": menu_name,
        "embedding": embedding,
        "related_menus": related_list
    }

    resp = client.index(index=index_name, body=doc)
    print(f"✅ 신메뉴 등록 완료: {menu_name}, id={resp['_id']}")
    print(f"➡️ related_menus: {related_list}")


# 🚀 실행
if __name__ == "__main__":
    add_new_menu_with_auto_rules("크림 까르보나라")
