#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_test.py
- 간단한 연관메뉴 테스트
"""

def test_related_menus_logic():
    """연관메뉴 생성 로직 테스트"""
    print("🧪 연관메뉴 생성 로직 테스트")
    
    # 시뮬레이션 데이터
    current_menu = {
        "menu_name": "크림 카르보나라",
        "category": "파스타",
        "description": "부드러운 크림소스와 베이컨이 어우러진 파스타"
    }
    
    other_menus = [
        {"menu_name": "로제 파스타", "category": "파스타", "description": "토마토 소스 파스타"},
        {"menu_name": "봉골레 파스타", "category": "파스타", "description": "해산물 파스타"},
        {"menu_name": "까르보나라", "category": "파스타", "description": "계란과 베이컨 파스타"},
        {"menu_name": "치즈돈까스", "category": "튀김", "description": "치즈가 들어간 돈까스"},
        {"menu_name": "콜라", "category": "음료", "description": "탄산음료"},
        {"menu_name": "크림 파스타", "category": "파스타", "description": "크림 소스 파스타"}
    ]
    
    print(f"현재 메뉴: {current_menu['menu_name']} ({current_menu['category']})")
    print("다른 메뉴들:")
    for menu in other_menus:
        print(f"  - {menu['menu_name']} ({menu['category']})")
    
    print("\n연관메뉴 생성 결과:")
    
    # 카테고리 보너스 계산
    for other_menu in other_menus:
        category_bonus = 2.0 if other_menu["category"] == current_menu["category"] else 1.0
        
        # 키워드 보너스 계산
        keyword_bonus = 1.0
        other_name = other_menu["menu_name"].lower()
        other_desc = other_menu.get("description", "").lower()
        
        if current_menu["category"] == "파스타":
            if any(keyword in other_name or keyword in other_desc 
                   for keyword in ["까르보", "카르보", "크림", "베이컨", "계란"]):
                keyword_bonus = 2.5
            elif any(keyword in other_name or keyword in other_desc 
                    for keyword in ["파스타", "면", "로제", "봉골레"]):
                keyword_bonus = 2.0
        
        final_bonus = category_bonus * keyword_bonus
        similarity = 0.7 * final_bonus  # 가상의 유사도
        
        print(f"  - {other_menu['menu_name']}: 유사도={similarity:.3f} (카테고리보너스:{category_bonus}x, 키워드보너스:{keyword_bonus}x)")

if __name__ == "__main__":
    test_related_menus_logic()
