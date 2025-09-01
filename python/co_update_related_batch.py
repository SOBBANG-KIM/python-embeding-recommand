import json
import itertools
from collections import defaultdict
from opensearchpy import OpenSearch

# ✅ OpenSearch 클라이언트
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

INDEX_NAME = "menus"
ORDERS_FILE = "../json/order_menu.jsonl"


def build_cooccurrence_matrix(orders):
    """주문 로그 기반 공동주문 행렬 생성"""
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


def update_related_menus(menu_name: str, related_list: list, index_name=INDEX_NAME):
    """특정 메뉴의 related_menus 업데이트"""
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


def main():
    # ✅ 주문 로그 불러오기
    orders = []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            orders.append(json.loads(line))

    # ✅ 공동주문 행렬 생성
    co_matrix, menu_counts = build_cooccurrence_matrix(orders)

    # ✅ 모든 메뉴별 related_menus 업데이트
    for menu in menu_counts.keys():
        related_list = []
        for other, count in co_matrix[menu].items():
            score = count / menu_counts[menu]  # 확률 기반 점수
            related_list.append({
                "menu_name": other,
                "co_occurrence_score": round(score, 3)  # 소수점 3자리
            })

        update_related_menus(menu, related_list)


if __name__ == "__main__":
    main()
