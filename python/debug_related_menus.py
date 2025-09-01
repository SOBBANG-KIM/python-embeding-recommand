#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_related_menus.py
- 연관메뉴 0개 문제 진단
"""

import json
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
)

def debug_index_contents(index_name: str = "menus"):
    """인덱스 내용 진단"""
    print(f"🔍 인덱스 '{index_name}' 내용 진단")
    print("="*50)
    
    # 전체 메뉴 조회
    resp = client.search(
        index=index_name,
        body={
            "size": 100,
            "query": {"match_all": {}},
            "_source": ["menu_name", "category", "related_menus", "embedding"]
        }
    )
    
    menus = resp["hits"]["hits"]
    print(f"📊 총 {len(menus)}개 메뉴 발견")
    print("---")
    
    for i, hit in enumerate(menus):
        menu = hit["_source"]
        related_count = len(menu.get("related_menus", []))
        has_embedding = "embedding" in menu and menu["embedding"]
        
        print(f"{i+1}. {menu['menu_name']} ({menu.get('category', 'N/A')})")
        print(f"   - 연관메뉴: {related_count}개")
        print(f"   - 임베딩: {'있음' if has_embedding else '없음'}")
        
        if related_count > 0:
            for item in menu["related_menus"][:3]:  # 상위 3개만
                print(f"     * {item.get('menu_name', 'N/A')} ({item.get('similarity_type', 'N/A')})")

def debug_specific_menu(menu_name: str, index_name: str = "menus"):
    """특정 메뉴 상세 진단"""
    print(f"🔍 '{menu_name}' 상세 진단")
    print("="*50)
    
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
    
    menu = resp["hits"]["hits"][0]["_source"]
    print(f"📋 메뉴명: {menu['menu_name']}")
    print(f"📋 카테고리: {menu.get('category', 'N/A')}")
    print(f"📋 설명: {menu.get('description', 'N/A')}")
    print(f"📋 임베딩: {'있음' if 'embedding' in menu and menu['embedding'] else '없음'}")
    print(f"📋 연관메뉴 수: {len(menu.get('related_menus', []))}")
    
    if menu.get("related_menus"):
        print("📋 연관메뉴 상세:")
        for i, item in enumerate(menu["related_menus"]):
            print(f"   {i+1}. {item}")

def debug_search_similar_menus(menu_name: str, index_name: str = "menus"):
    """유사 메뉴 검색 테스트"""
    print(f"🔍 '{menu_name}' 유사 메뉴 검색 테스트")
    print("="*50)
    
    # 1. 해당 메뉴의 임베딩 가져오기
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
    
    menu = resp["hits"]["hits"][0]["_source"]
    embedding = menu.get("embedding")
    category = menu.get("category")
    
    if not embedding:
        print("❌ 임베딩이 없습니다.")
        return
    
    print(f"✅ 임베딩 발견: {len(embedding)}차원")
    print(f"✅ 카테고리: {category}")
    
    # 2. 같은 카테고리에서 다른 메뉴 찾기
    print("\n🔍 같은 카테고리 메뉴 검색:")
    category_query = {
        "bool": {
            "must": [
                {"term": {"category": category}},
                {"bool": {"must_not": [{"term": {"menu_name": menu_name}}]}}
            ]
        }
    }
    
    resp = client.search(
        index=index_name,
        body={
            "size": 10,
            "query": category_query,
            "_source": ["menu_name", "category", "embedding"]
        }
    )
    
    similar_menus = resp["hits"]["hits"]
    print(f"📊 같은 카테고리 메뉴: {len(similar_menus)}개")
    
    for hit in similar_menus:
        other_menu = hit["_source"]
        print(f"   - {other_menu['menu_name']} (임베딩: {'있음' if 'embedding' in other_menu else '없음'})")
    
    # 3. 전체 메뉴에서 검색
    print("\n🔍 전체 메뉴 검색:")
    resp = client.search(
        index=index_name,
        body={
            "size": 10,
            "query": {
                "bool": {
                    "must_not": [
                        {"term": {"menu_name": menu_name}}
                    ]
                }
            },
            "_source": ["menu_name", "category", "embedding"]
        }
    )
    
    all_menus = resp["hits"]["hits"]
    print(f"📊 전체 메뉴: {len(all_menus)}개")
    
    for hit in all_menus:
        other_menu = hit["_source"]
        print(f"   - {other_menu['menu_name']} ({other_menu.get('category', 'N/A')}) (임베딩: {'있음' if 'embedding' in other_menu else '없음'})")

if __name__ == "__main__":
    # 1. 인덱스 전체 내용 확인
    debug_index_contents()
    print("\n" + "="*50 + "\n")
    
    # 2. 크림 카르보나라 상세 진단
    debug_specific_menu("크림 카르보나라")
    print("\n" + "="*50 + "\n")
    
    # 3. 유사 메뉴 검색 테스트
    debug_search_similar_menus("크림 카르보나라")
