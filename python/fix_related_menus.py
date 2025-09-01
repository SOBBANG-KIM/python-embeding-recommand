#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_related_menus.py
- 연관메뉴 0개 문제 해결
- 강제로 연관메뉴를 생성하는 스크립트
"""

import json
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from opensearchpy import OpenSearch
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


def force_create_related_menus():
    """강제로 모든 메뉴에 연관메뉴 생성"""
    print("🔧 강제 연관메뉴 생성 시작")
    
    # 1. 모든 메뉴 조회
    resp = client.search(
        index=INDEX_NAME,
        body={
            "size": 1000,
            "query": {"match_all": {}},
            "_source": ["menu_name", "category", "description", "embedding"]
        }
    )
    
    menus = resp["hits"]["hits"]
    print(f"📊 총 {len(menus)}개 메뉴 발견")
    
    if len(menus) < 2:
        print("❌ 메뉴가 2개 미만입니다. 연관메뉴를 생성할 수 없습니다.")
        return
    
    # 2. 메뉴별 임베딩과 카테고리 정보 수집
    menu_data = {}
    for hit in menus:
        menu = hit["_source"]
        menu_id = hit["_id"]
        menu_data[menu_id] = {
            "menu_name": menu.get("menu_name", ""),
            "category": menu.get("category", ""),
            "description": menu.get("description", ""),
            "embedding": menu.get("embedding", [])
        }
    
    # 3. 각 메뉴에 대해 연관메뉴 생성
    for menu_id, current_menu in menu_data.items():
        print(f"\n🔍 {current_menu['menu_name']} 연관메뉴 생성 중...")
        
        related_list = []
        current_embedding = current_menu["embedding"]
        current_category = current_menu["category"]
        
        if not current_embedding:
            print(f"   ❌ {current_menu['menu_name']}에 임베딩이 없습니다.")
            continue
        
        # 3-1. 같은 카테고리에서 유사한 메뉴 찾기
        same_category_menus = []
        for other_id, other_menu in menu_data.items():
            if other_id == menu_id:
                continue
            
            if other_menu["category"] == current_category:
                if len(other_menu["embedding"]) > 0:
                    # 코사인 유사도 계산
                    similarity = np.dot(current_embedding, other_menu["embedding"]) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(other_menu["embedding"])
                    )
                    
                    # 키워드 보너스
                    keyword_bonus = 1.0
                    other_name = other_menu["menu_name"].lower()
                    other_desc = other_menu.get("description", "").lower()
                    
                    # 카테고리별 키워드 보너스
                    if current_category == "파스타":
                        if any(keyword in other_name or keyword in other_desc 
                               for keyword in ["까르보", "카르보", "크림", "베이컨", "계란"]):
                            keyword_bonus = 2.0
                        elif any(keyword in other_name or keyword in other_desc 
                                for keyword in ["파스타", "면", "로제", "봉골레"]):
                            keyword_bonus = 1.5
                    elif current_category == "치킨":
                        if any(keyword in other_name or keyword in other_desc 
                               for keyword in ["치킨", "닭", "튀김"]):
                            keyword_bonus = 1.5
                    elif current_category == "음료":
                        if any(keyword in other_name or keyword in other_desc 
                               for keyword in ["콜라", "사이다", "탄산"]):
                            keyword_bonus = 1.5
                    
                    adjusted_similarity = similarity * keyword_bonus
                    
                    same_category_menus.append({
                        "menu_name": other_menu["menu_name"],
                        "similarity": adjusted_similarity,
                        "category": other_menu["category"]
                    })
        
        # 3-2. 다른 카테고리에서도 유사한 메뉴 찾기
        other_category_menus = []
        for other_id, other_menu in menu_data.items():
            if other_id == menu_id:
                continue
            
            if other_menu["category"] != current_category:
                if len(other_menu["embedding"]) > 0:
                    # 코사인 유사도 계산
                    similarity = np.dot(current_embedding, other_menu["embedding"]) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(other_menu["embedding"])
                    )
                    
                    # 키워드 보너스 (낮은 보너스)
                    keyword_bonus = 1.0
                    other_name = other_menu["menu_name"].lower()
                    other_desc = other_menu.get("description", "").lower()
                    
                    if any(keyword in other_name or keyword in other_desc 
                           for keyword in ["크림", "소스", "베이컨", "계란", "치즈"]):
                        keyword_bonus = 1.2
                    
                    adjusted_similarity = similarity * keyword_bonus
                    
                    if adjusted_similarity > 0.2:  # 낮은 임계값
                        other_category_menus.append({
                            "menu_name": other_menu["menu_name"],
                            "similarity": adjusted_similarity,
                            "category": other_menu["category"]
                        })
        
        # 3-3. 최종 연관메뉴 리스트 구성
        # 같은 카테고리 우선 (상위 3개)
        same_category_menus.sort(key=lambda x: x["similarity"], reverse=True)
        for item in same_category_menus[:3]:
            related_list.append({
                "menu_name": item["menu_name"],
                "embedding_similarity": round(item["similarity"], 3),
                "similarity_type": "category_similarity",
                "category": item["category"]
            })
        
        # 다른 카테고리 보완 (상위 2개)
        other_category_menus.sort(key=lambda x: x["similarity"], reverse=True)
        for item in other_category_menus[:2]:
            related_list.append({
                "menu_name": item["menu_name"],
                "embedding_similarity": round(item["similarity"], 3),
                "similarity_type": "cross_category_similarity",
                "category": item["category"]
            })
        
        # 3-4. 연관메뉴가 없으면 랜덤하게 추가
        if not related_list:
            print(f"   ⚠️ {current_menu['menu_name']}에 연관메뉴가 없어 랜덤 추가")
            other_menus = [m for m in menu_data.values() if m["menu_name"] != current_menu["menu_name"]]
            for other_menu in other_menus[:3]:
                related_list.append({
                    "menu_name": other_menu["menu_name"],
                    "embedding_similarity": 0.5,  # 기본값
                    "similarity_type": "random_fallback",
                    "category": other_menu["category"]
                })
        
        # 3-5. OpenSearch 업데이트
        if related_list:
            client.update(
                index=INDEX_NAME,
                id=menu_id,
                body={
                    "doc": {
                        "related_menus": related_list,
                        "updated_at": now_iso()
                    }
                }
            )
            print(f"   ✅ {len(related_list)}개 연관메뉴 생성 완료")
            for item in related_list[:3]:  # 상위 3개만 출력
                print(f"      - {item['menu_name']} ({item['similarity_type']})")
        else:
            print(f"   ❌ {current_menu['menu_name']} 연관메뉴 생성 실패")
    
    print("\n🎉 강제 연관메뉴 생성 완료!")


def verify_related_menus():
    """연관메뉴 생성 결과 확인"""
    print("\n🔍 연관메뉴 생성 결과 확인")
    print("="*50)
    
    resp = client.search(
        index=INDEX_NAME,
        body={
            "size": 100,
            "query": {"match_all": {}},
            "_source": ["menu_name", "category", "related_menus"]
        }
    )
    
    menus = resp["hits"]["hits"]
    total_menus = len(menus)
    menus_with_related = 0
    
    for hit in menus:
        menu = hit["_source"]
        related_count = len(menu.get("related_menus", []))
        
        if related_count > 0:
            menus_with_related += 1
            print(f"✅ {menu['menu_name']}: {related_count}개 연관메뉴")
        else:
            print(f"❌ {menu['menu_name']}: 연관메뉴 없음")
    
    print(f"\n📊 결과: {menus_with_related}/{total_menus} 메뉴에 연관메뉴 생성됨")


if __name__ == "__main__":
    try:
        # 1. 강제 연관메뉴 생성
        force_create_related_menus()
        
        # 2. 결과 확인
        verify_related_menus()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
