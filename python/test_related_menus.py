#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_related_menus.py
- 연관메뉴 기능 테스트 스크립트
- 크림 카르보나라의 연관메뉴 확인
"""

import json
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

def test_related_menus(menu_name: str, index_name: str = "menus"):
    """특정 메뉴의 연관메뉴 확인"""
    print(f"🔍 '{menu_name}' 연관메뉴 확인")
    
    # 메뉴 검색
    resp = client.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "menu_name": menu_name
                }
            }
        }
    )
    
    if not resp["hits"]["hits"]:
        print(f"❌ '{menu_name}' 메뉴를 찾을 수 없습니다.")
        return
    
    menu_doc = resp["hits"]["hits"][0]["_source"]
    related_menus = menu_doc.get("related_menus", [])
    
    print(f"📋 메뉴: {menu_doc['menu_name']}")
    print(f"📋 카테고리: {menu_doc.get('category', 'N/A')}")
    print(f"📋 연관메뉴 수: {len(related_menus)}개")
    print("---")
    
    if not related_menus:
        print("❌ 연관메뉴가 없습니다.")
        return
    
    # 연관메뉴 분류
    co_occurrence_items = [item for item in related_menus if item.get("similarity_type") == "co_occurrence"]
    embedding_items = [item for item in related_menus if item.get("similarity_type") == "embedding"]
    
    print(f"🔄 공동주문 기반: {len(co_occurrence_items)}개")
    for item in co_occurrence_items:
        print(f"   - {item['menu_name']} (점수: {item.get('co_occurrence_score', 'N/A')})")
    
    print(f"🧠 임베딩 유사도 기반: {len(embedding_items)}개")
    for item in embedding_items:
        print(f"   - {item['menu_name']} (유사도: {item.get('embedding_similarity', 'N/A')}, 카테고리: {item.get('category', 'N/A')})")

def test_all_menus_related_info(index_name: str = "menus"):
    """모든 메뉴의 연관메뉴 정보 확인"""
    print(f"📊 전체 메뉴 연관메뉴 정보 확인")
    
    resp = client.search(
        index=index_name,
        body={
            "size": 100,
            "query": {"match_all": {}},
            "_source": ["menu_name", "category", "related_menus"]
        }
    )
    
    menus = resp["hits"]["hits"]
    print(f"📋 총 {len(menus)}개 메뉴 발견")
    print("---")
    
    for hit in menus:
        menu = hit["_source"]
        related_count = len(menu.get("related_menus", []))
        co_occurrence_count = len([item for item in menu.get("related_menus", []) 
                                 if item.get("similarity_type") == "co_occurrence"])
        embedding_count = len([item for item in menu.get("related_menus", []) 
                             if item.get("similarity_type") == "embedding"])
        
        print(f"🍽️ {menu['menu_name']} ({menu.get('category', 'N/A')})")
        print(f"   - 총 연관메뉴: {related_count}개")
        print(f"   - 공동주문: {co_occurrence_count}개")
        print(f"   - 임베딩 유사도: {embedding_count}개")

if __name__ == "__main__":
    # 크림 카르보나라 테스트
    test_related_menus("크림 카르보나라")
    print("\n" + "="*50 + "\n")
    
    # 전체 메뉴 정보 확인
    test_all_menus_related_info()
