#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_carbonara.py
- 크림 카르보나라 연관메뉴 테스트
"""

from add_menu_with_auto_rules2 import add_new_menu_with_auto_rules
from test_related_menus import test_related_menus

def test_carbonara():
    print("🍝 크림 카르보나라 연관메뉴 테스트 시작")
    print("="*50)
    
    # 1. 크림 카르보나라 추가
    print("1️⃣ 크림 카르보나라 메뉴 추가 중...")
    add_new_menu_with_auto_rules(
        menu_name="크림 카르보나라",
        description="부드러운 크림소스와 베이컨이 어우러진 파스타",
        category="파스타",
        price=15000,
        w_pop=1.2,
        w_recency=1.0,
        w_custom=1.1
    )
    
    print("\n" + "="*50)
    
    # 2. 연관메뉴 확인
    print("2️⃣ 연관메뉴 확인 중...")
    test_related_menus("크림 카르보나라")

if __name__ == "__main__":
    test_carbonara()
