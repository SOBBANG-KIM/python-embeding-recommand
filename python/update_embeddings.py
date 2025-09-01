#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_embeddings.py
- 신규메뉴 추가 시 기존 메뉴들의 임베딩을 재계산하여 벡터 공간 업데이트
- 새로운 메뉴 정보를 반영하여 더 정확한 유사도 계산 가능

실행 예시:
  python update_embeddings.py --index menus --batch-size 32
"""

import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np

# 설정
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


def get_all_menus_from_index(index_name: str) -> List[Dict[str, Any]]:
    """인덱스에서 모든 메뉴 데이터 조회"""
    menus = []
    
    # Scroll API로 모든 문서 조회
    resp = client.search(
        index=index_name,
        body={
            "size": 1000,  # 한 번에 가져올 문서 수
            "query": {"match_all": {}},
            "_source": ["menu_id", "menu_name", "category", "price", "description", 
                       "w_pop", "w_recency", "w_custom", "related_menus"]
        },
        scroll="5m"
    )
    
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    
    while hits:
        for hit in hits:
            source = hit["_source"]
            source["_id"] = hit["_id"]  # 문서 ID 추가
            menus.append(source)
        
        # 다음 배치 조회
        resp = client.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
    
    # Scroll 정리
    client.clear_scroll(scroll_id=scroll_id)
    
    return menus


def update_embeddings_full_reindex(index_name: str, batch_size: int = 32):
    """전체 메뉴의 임베딩을 재계산하여 인덱스 업데이트"""
    print(f"🔄 전체 임베딩 재계산 시작: {index_name}")
    
    # 1. 모든 메뉴 데이터 조회
    menus = get_all_menus_from_index(index_name)
    print(f"📊 총 {len(menus)}개 메뉴 발견")
    
    if not menus:
        print("❌ 업데이트할 메뉴가 없습니다.")
        return
    
    # 2. 임베딩 텍스트 생성
    embed_texts = []
    for menu in menus:
        embed_text = build_embed_text(
            menu.get("menu_name", ""),
            menu.get("description", ""),
            menu.get("category", ""),
            to_float(menu.get("price", 0))
        )
        embed_texts.append(embed_text)
    
    # 3. 새로운 임베딩 생성
    print(f"🧠 {len(embed_texts)}개 메뉴 임베딩 생성 중...")
    embeddings = model.encode(
        embed_texts, 
        batch_size=batch_size,
        convert_to_numpy=True, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    
    # 4. 벌크 업데이트
    def gen_update_actions():
        for menu, embedding in zip(menus, embeddings):
            yield {
                "_index": index_name,
                "_id": menu["_id"],
                "_op_type": "update",
                "doc": {
                    "embedding": embedding.astype(float).tolist(),
                    "updated_at": now_iso()
                }
            }
    
    print("📝 벌크 업데이트 중...")
    success, failed = helpers.bulk(
        client, 
        gen_update_actions(), 
        chunk_size=1000, 
        request_timeout=120, 
        stats_only=True
    )
    
    print(f"✅ 임베딩 업데이트 완료: 성공={success}, 실패={failed}")


def update_embeddings_incremental(new_menus: List[Dict[str, Any]], index_name: str, batch_size: int = 32):
    """새로운 메뉴들만 추가하여 임베딩 공간 확장"""
    print(f"🆕 증분 임베딩 업데이트 시작: {len(new_menus)}개 신규메뉴")
    
    if not new_menus:
        print("❌ 추가할 신규메뉴가 없습니다.")
        return
    
    # 1. 임베딩 텍스트 생성
    embed_texts = []
    for menu in new_menus:
        embed_text = build_embed_text(
            menu.get("menu_name", ""),
            menu.get("description", ""),
            menu.get("category", ""),
            to_float(menu.get("price", 0))
        )
        embed_texts.append(embed_text)
    
    # 2. 새로운 임베딩 생성
    print(f"🧠 {len(embed_texts)}개 신규메뉴 임베딩 생성 중...")
    embeddings = model.encode(
        embed_texts, 
        batch_size=batch_size,
        convert_to_numpy=True, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    
    # 3. 신규메뉴 추가
    def gen_index_actions():
        for menu, embedding in zip(new_menus, embeddings):
            doc = {
                "menu_id": menu.get("menu_id"),
                "menu_name": menu.get("menu_name"),
                "category": menu.get("category"),
                "price": to_float(menu.get("price", 0)),
                "description": menu.get("description", ""),
                "created_at": now_iso(),
                "embedding": embedding.astype(float).tolist(),
                "related_menus": menu.get("related_menus", []),
                "w_pop": to_float(menu.get("w_pop", 1.0)),
                "w_recency": to_float(menu.get("w_recency", 1.0)),
                "w_custom": to_float(menu.get("w_custom", 1.0)),
                "attributes": derive_attrs(
                    menu.get("menu_name", ""),
                    menu.get("description", ""),
                    menu.get("category", "")
                )
            }
            
            yield {
                "_index": index_name,
                "_id": menu.get("menu_id"),
                "_op_type": "index",
                "_source": doc
            }
    
    print("📝 신규메뉴 인덱싱 중...")
    success, failed = helpers.bulk(
        client, 
        gen_index_actions(), 
        chunk_size=1000, 
        request_timeout=120, 
        stats_only=True
    )
    
    print(f"✅ 신규메뉴 추가 완료: 성공={success}, 실패={failed}")


def update_embeddings_with_related_menus(index_name: str, batch_size: int = 32, include_embedding_similarity: bool = True):
    """연관메뉴 정보를 포함하여 임베딩 업데이트 (임베딩 유사도 포함)"""
    print(f"🔗 연관메뉴 정보 포함 임베딩 업데이트 시작")
    
    # 1. 주문 로그 읽기
    orders = []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            orders.append(json.loads(line))
    
    # 2. 공동주문 매트릭스 계산
    from collections import defaultdict
    import itertools
    
    co_matrix = defaultdict(lambda: defaultdict(int))
    menu_counts = defaultdict(int)
    
    for order in orders:
        menus = order["menus"]
        for menu in menus:
            menu_counts[menu] += 1
        for a, b in itertools.combinations(sorted(menus), 2):
            co_matrix[a][b] += 1
            co_matrix[b][a] += 1
    
    # 3. 모든 메뉴 조회
    menus = get_all_menus_from_index(index_name)
    print(f"📊 총 {len(menus)}개 메뉴 발견")
    
    # 4. 임베딩 유사도 계산을 위한 임베딩 생성
    if include_embedding_similarity:
        print("🧠 임베딩 유사도 계산을 위한 텍스트 생성 중...")
        menu_embeddings = {}
        
        for menu in menus:
            embed_text = build_embed_text(
                menu.get("menu_name", ""),
                menu.get("description", ""),
                menu.get("category", ""),
                to_float(menu.get("price", 0))
            )
            embedding = model.encode(embed_text)
            menu_embeddings[menu["_id"]] = {
                "embedding": embedding,
                "menu_name": menu.get("menu_name", ""),
                "category": menu.get("category", "")
            }
    
    # 5. 연관메뉴 정보 업데이트
    def gen_update_actions():
        for menu in menus:
            menu_name = menu.get("menu_name", "")
            menu_id = menu["_id"]
            related_list = []
            
            # 공동주문 기반 연관메뉴
            if menu_name in co_matrix:
                for other, count in co_matrix[menu_name].items():
                    score = count / menu_counts[menu_name]
                    related_list.append({
                        "menu_name": other,
                        "co_occurrence_score": round(score, 3),
                        "similarity_type": "co_occurrence"
                    })
            
            # 임베딩 유사도 기반 연관메뉴 (공동주문 데이터가 없거나 부족한 경우)
            if include_embedding_similarity and menu_id in menu_embeddings:
                current_embedding = menu_embeddings[menu_id]["embedding"]
                current_category = menu_embeddings[menu_id]["category"]
                
                # 같은 카테고리 내에서 유사한 메뉴 찾기
                similar_menus = []
                for other_id, other_data in menu_embeddings.items():
                    if other_id == menu_id:
                        continue
                    
                    # 같은 카테고리 우선, 다른 카테고리도 허용
                    category_bonus = 1.5 if other_data["category"] == current_category else 1.0
                    
                    # 코사인 유사도 계산
                    similarity = np.dot(current_embedding, other_data["embedding"]) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(other_data["embedding"])
                    )
                    
                    # 카테고리 보너스 적용
                    adjusted_similarity = similarity * category_bonus
                    
                    if adjusted_similarity > 0.3:  # 임계값
                        similar_menus.append({
                            "menu_name": other_data["menu_name"],
                            "embedding_similarity": round(adjusted_similarity, 3),
                            "category": other_data["category"]
                        })
                
                # 유사도 순으로 정렬하고 상위 5개 선택
                similar_menus.sort(key=lambda x: x["embedding_similarity"], reverse=True)
                
                for similar_menu in similar_menus[:5]:
                    # 공동주문 데이터가 없는 경우에만 임베딩 유사도 추가
                    existing_co_occurrence = any(
                        item["menu_name"] == similar_menu["menu_name"] 
                        for item in related_list
                    )
                    
                    if not existing_co_occurrence:
                        related_list.append({
                            "menu_name": similar_menu["menu_name"],
                            "embedding_similarity": similar_menu["embedding_similarity"],
                            "similarity_type": "embedding",
                            "category": similar_menu["category"]
                        })
            
            # 최종 연관메뉴 리스트 (공동주문 우선, 임베딩 유사도 보완)
            final_related_list = []
            
            # 공동주문 기반 먼저 추가
            co_occurrence_items = [item for item in related_list if item.get("similarity_type") == "co_occurrence"]
            final_related_list.extend(co_occurrence_items)
            
            # 임베딩 유사도 기반 추가 (공동주문이 부족한 경우)
            embedding_items = [item for item in related_list if item.get("similarity_type") == "embedding"]
            final_related_list.extend(embedding_items[:max(0, 5 - len(co_occurrence_items))])
            
            print(f"📋 {menu_name}: 공동주문 {len(co_occurrence_items)}개, 임베딩 유사도 {len(embedding_items)}개")
            
            yield {
                "_index": index_name,
                "_id": menu_id,
                "_op_type": "update",
                "doc": {
                    "related_menus": final_related_list,
                    "updated_at": now_iso()
                }
            }
    
    print("📝 연관메뉴 정보 업데이트 중...")
    success, failed = helpers.bulk(
        client, 
        gen_update_actions(), 
        chunk_size=1000, 
        request_timeout=120, 
        stats_only=True
    )
    
    print(f"✅ 연관메뉴 정보 업데이트 완료: 성공={success}, 실패={failed}")


def main():
    parser = argparse.ArgumentParser(description="메뉴 임베딩 업데이트 도구")
    parser.add_argument("--index", default=INDEX_NAME, help="인덱스 이름")
    parser.add_argument("--mode", choices=["full", "incremental", "related"], 
                       default="full", help="업데이트 모드")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--new-menus-file", help="신규메뉴 JSON 파일 (incremental 모드용)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        update_embeddings_full_reindex(args.index, args.batch_size)
    elif args.mode == "incremental":
        if not args.new_menus_file:
            print("❌ incremental 모드에서는 --new-menus-file이 필요합니다.")
            return
        
        with open(args.new_menus_file, "r", encoding="utf-8") as f:
            new_menus = json.load(f)
        update_embeddings_incremental(new_menus, args.index, args.batch_size)
    elif args.mode == "related":
        update_embeddings_with_related_menus(args.index, args.batch_size)


if __name__ == "__main__":
    main()
