import json
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_NAME = "menus"
ORDERS_FILE = "../json/order_menu.jsonl"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


# ---- 텍스트 전처리 유틸 ----
def price_bucket(won: float) -> str:
    if won <= 5000: return "저가"
    if won <= 12000: return "중가"
    return "고가"

def derive_attrs(name: str, desc: str, category: str) -> list[str]:
    txt = (name + " " + desc).lower()
    attrs = []

    # 맛/조리
    if any(k in txt for k in ["매콤","매운","얼큰"]): attrs += ["맛: 매움"]
    if "달콤" in txt: attrs += ["맛: 달콤"]
    if any(k in txt for k in ["담백","깔끔"]): attrs += ["맛: 담백"]
    if any(k in txt for k in ["토마토","로제"]): attrs += ["소스: 토마토/로제"]
    if any(k in txt for k in ["크림","까르보"]): attrs += ["소스: 크림"]
    if any(k in txt for k in ["볶음","구이","튀김"]): attrs += ["조리: 볶음/구이/튀김"]
    if any(k in txt for k in ["국","국밥","탕","찌개"]): attrs += ["형태: 국/탕/찌개"]

    # 재료/종류
    if any(k in txt for k in ["해물","해산물","새우","오징어","문어","홍합"]): attrs += ["주재료: 해산물"]
    if any(k in txt for k in ["소고기","쇠고기","우","차돌"]): attrs += ["주재료: 소고기"]
    if any(k in txt for k in ["돼지","삼겹","목살","돈"]): attrs += ["주재료: 돼지고기"]
    if any(k in txt for k in ["닭","치킨","계"]): attrs += ["주재료: 닭고기"]
    if any(k in txt for k in ["면","파스타","라면","우동"]): attrs += ["주식: 면"]
    if any(k in txt for k in ["밥","비빔밥","덮밥","볶음밥"]): attrs += ["주식: 밥"]

    # 음료/주류 보강
    if "콜라" in txt: attrs += ["음료: 탄산", "맛: 달콤", "차가움"]
    if any(k in txt for k in ["사이다","스프라이트"]): attrs += ["음료: 탄산", "차가움"]
    if any(k in txt for k in ["맥주","소주","참이슬","청하","위스키","와인"]): attrs += ["주류"]

    # 카테고리 자체를 태그로
    if category:
        attrs += [f"카테고리: {category}"]

    return attrs

def build_embed_text(menu_name: str, description: str = "", category: str = "", price: float = 0) -> str:
    name = menu_name.strip()
    desc = description.strip()
    cat = category.strip()
    attrs = derive_attrs(name, desc, cat)
    attrs.append(f"가격대: {price_bucket(price)}")
    # 최종 문장
    return ". ".join(filter(None, [name, desc] + attrs)) + "."


def ensure_index_with_weights(client: OpenSearch, index_name: str, dim: int = 384):
    """가중치 필드가 포함된 인덱스 생성"""
    if client.indices.exists(index=index_name):
        return
    
    mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "menu_id": {"type": "keyword"},
                "menu_name": {"type": "text"},
                "category": {"type": "keyword"},
                "price": {"type": "float"},
                "description": {"type": "text"},
                "created_at": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 24, "ef_construction": 128},
                    },
                },
                "related_menus": {
                    "type": "nested",
                    "properties": {
                        "menu_name": {"type": "keyword"},
                        "co_occurrence_score": {"type": "float"}
                    }
                },
                # 가중치 필드들
                "w_pop": {"type": "float"},
                "w_recency": {"type": "float"},
                "w_custom": {"type": "float"},
                "attributes": {"type": "keyword"},  # 자동 추출된 속성들
            }
        },
    }
    client.indices.create(index=index_name, body=mapping)
    print(f"✅ 인덱스 생성 완료: {index_name}")


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


def add_new_menu_with_auto_rules(
    menu_name: str, 
    description: str = "", 
    category: str = "", 
    price: float = 0.0,
    menu_id: str = None,
    w_pop: float = 1.0,
    w_recency: float = 1.0,
    w_custom: float = 1.0,
    index_name=INDEX_NAME
):
    # ✅ 1. 주문 로그 읽기
    orders = []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            orders.append(json.loads(line))

    co_matrix, menu_counts = build_cooccurrence_matrix(orders)

    # ✅ 2. 인덱스 확인/생성
    ensure_index_with_weights(client, index_name)

    # ✅ 3. 텍스트 전처리 및 임베딩
    embed_text = build_embed_text(menu_name, description, category, price)
    embedding = model.encode(embed_text).tolist()
    
    # ✅ 4. 속성 추출
    attributes = derive_attrs(menu_name, description, category)

    # ✅ 5. 공동주문 데이터 있으면 직접 반영
    related_list = []
    if menu_name in co_matrix:
        for other, count in co_matrix[menu_name].items():
            score = count / menu_counts[menu_name]
            related_list.append({
                "menu_name": other,
                "co_occurrence_score": round(score, 3)
            })

    # ✅ 6. 공동주문 데이터가 없으면 → 임베딩 기반 유사 메뉴 찾기
    if not related_list:
        print("📌 공동주문 데이터 없음 → 임베딩 유사도 기반 연관메뉴 생성")
        
        # 임베딩 유사도 기반으로 유사한 메뉴들 찾기
        resp = client.search(
            index=index_name,
            body={
                "size": 10,  # 더 많은 후보 검색
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": 10
                        }
                    }
                },
                "_source": ["menu_name", "category", "embedding"]
            }
        )
        
        if resp["hits"]["hits"]:
            # 코사인 유사도 계산
            current_embedding = np.array(embedding)
            similar_menus = []
            
            for hit in resp["hits"]["hits"]:
                other_menu = hit["_source"]
                other_embedding = np.array(other_menu["embedding"])
                
                # 코사인 유사도 계산
                similarity = np.dot(current_embedding, other_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                )
                
                # 카테고리 보너스 (같은 카테고리 우선)
                category_bonus = 1.5 if other_menu.get("category") == category else 1.0
                adjusted_similarity = similarity * category_bonus
                
                if adjusted_similarity > 0.3:  # 임계값
                    similar_menus.append({
                        "menu_name": other_menu["menu_name"],
                        "embedding_similarity": round(adjusted_similarity, 3),
                        "similarity_type": "embedding",
                        "category": other_menu.get("category", "")
                    })
            
            # 유사도 순으로 정렬하고 상위 5개 선택
            similar_menus.sort(key=lambda x: x["embedding_similarity"], reverse=True)
            related_list = similar_menus[:5]
            
            print(f"🔍 임베딩 유사도 기반 연관메뉴 {len(related_list)}개 생성")
            for item in related_list:
                print(f"   - {item['menu_name']} (유사도: {item['embedding_similarity']})")

    # ✅ 7. OpenSearch에 저장 (가중치 필드 포함)
    doc = {
        "menu_id": menu_id or f"menu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "menu_name": menu_name,
        "category": category,
        "price": price,
        "description": description,
        "created_at": now_iso(),
        "embedding": embedding,
        "related_menus": related_list,
        # 가중치 필드들
        "w_pop": w_pop,
        "w_recency": w_recency,
        "w_custom": w_custom,
        "attributes": attributes
    }

    resp = client.index(index=index_name, body=doc)
    print(f"✅ 신메뉴 등록 완료: {menu_name}, id={resp['_id']}")
    print(f"➡️ 가격대: {price_bucket(price)}")
    print(f"➡️ 속성: {attributes}")
    print(f"➡️ 연관메뉴: {len(related_list)}개")
    return resp['_id']


# 🚀 실행 예시
if __name__ == "__main__":
    # 기본 예시
    # add_new_menu_with_auto_rules("로제 파스타")
    
    # 상세 정보 포함 예시
    add_new_menu_with_auto_rules(
        menu_name="크림 카르보나라",
        description="부드러운 크림소스와 베이컨이 어우러진 파스타",
        category="파스타",
        price=15000,
        w_pop=1.2,  # 인기도 높음
        w_recency=1.0,
        w_custom=1.1
    )
