\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
menu_embed.py
- 로컬(Mac, 보안 OFF) OpenSearch에 음식 메뉴 데이터를 임베딩 생성 후 벌크 인덱싱
- 입력: JSONL 또는 CSV (필수 컬럼: menu_id, menu_name, category, price, description, [created_at])
- 기본 모델: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384차원, 한국어 OK)

설치:
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install "sentence-transformers>=2.7.0" "opensearch-py>=2.7.1" "pandas>=2.2.0" "tqdm>=4.66.0"

실행 예시:
  python menu_embed.py --input menus.jsonl --index menu_items_v1 --alias menu_items \
    --host http://localhost:9200 --batch-size 64

CSV 예시 헤더:
  menu_id,menu_name,category,price,description,created_at

주의:
- OpenSearch가 보안 OFF로 9200에서 떠 있어야 함.
- 최초 실행 시 모델 다운로드에 네트워크가 필요.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("sentence-transformers 가 설치되어 있지 않습니다. pip install sentence-transformers", file=sys.stderr)
    raise

try:
    from opensearchpy import OpenSearch, helpers, RequestsHttpConnection
except Exception as e:
    print("opensearch-py 가 설치되어 있지 않습니다. pip install opensearch-py", file=sys.stderr)
    raise


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def read_csv(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            docs.append(row)
    return docs


def ensure_index(client: OpenSearch, index: str, dim: int):
    if client.indices.exists(index=index):
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
                "text_vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 24, "ef_construction": 128},
                    },
                },
                "w_pop": {"type": "float"},
                "w_recency": {"type": "float"},
                "w_custom": {"type": "float"},
            }
        },
    }
    client.indices.create(index=index, body=mapping)


def ensure_alias(client: OpenSearch, index: str, alias: str):
    if not alias:
        return
    # 이미 alias가 index를 가리키면 패스
    if client.indices.exists_alias(name=alias, index=index):
        return
    # alias가 다른 인덱스를 가리키고 있으면 교체
    if client.indices.exists_alias(name=alias):
        ali = client.indices.get_alias(name=alias)
        actions = []
        for idx in ali.keys():
            actions.append({"remove": {"index": idx, "alias": alias}})
        actions.append({"add": {"index": index, "alias": alias}})
        client.indices.update_aliases({"actions": actions})
    else:
        client.indices.put_alias(index=index, name=alias)


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def build_source(row: Dict[str, Any], emb: np.ndarray) -> Dict[str, Any]:
    menu_id = row.get("menu_id") or row.get("id")
    menu_name = (row.get("menu_name") or "").strip()
    category = (row.get("category") or "").strip()
    price = to_float(row.get("price", 0))
    desc = (row.get("description") or "").strip()
    created_at = row.get("created_at") or now_iso()

    src = {
        "menu_id": menu_id,
        "menu_name": menu_name,
        "category": category,
        "price": price,
        "description": desc,
        "created_at": created_at,
        "text_vector": emb.astype(float).tolist(),
        # 기본 가중치
        "w_pop": to_float(row.get("w_pop", 1.0), 1.0),
        "w_recency": to_float(row.get("w_recency", 1.0), 1.0),
        "w_custom": to_float(row.get("w_custom", 1.0), 1.0),
    }
    return src


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

def build_embed_text(row: dict) -> str:
    name = (row.get("menu_name") or "").strip()
    desc = (row.get("description") or "").strip()
    cat  = (row.get("category") or "").strip()
    price = float(row.get("price") or 0)
    attrs = derive_attrs(name, desc, cat)
    attrs.append(f"가격대: {price_bucket(price)}")
    # 최종 문장
    return ". ".join(filter(None, [name, desc] + attrs)) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 파일 경로 (.jsonl 또는 .csv)")
    ap.add_argument("--index", default="menu_items_v1", help="인덱스 이름(기본: menu_items_v1)")
    ap.add_argument("--alias", default="menu_items", help="별칭(alias) 이름(공란이면 비활성)")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch 호스트 URL")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="Sentence-Transformers 모델 이름 또는 로컬 경로")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="임베딩만 생성하고 인덱싱은 하지 않음")
    args = ap.parse_args()

    # OpenSearch 클라이언트 (보안 OFF 가정)
    client = OpenSearch(
        hosts=[args.host],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

    # 데이터 로드
    if args.input.lower().endswith(".jsonl"):
        rows = read_jsonl(args.input)
    elif args.input.lower().endswith(".csv"):
        rows = read_csv(args.input)
    else:
        print("지원하지 않는 포맷입니다. .jsonl 또는 .csv 를 사용하세요.", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("입력 데이터가 비어 있습니다.", file=sys.stderr)
        sys.exit(1)

    # 모델 로드
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension = {dim}")

    # 인덱스 준비
    if not args.dry_run:
        ensure_index(client, args.index, dim)
        ensure_alias(client, args.index, args.alias)

    # 텍스트 만들기
    texts = [build_embed_text(r) for r in rows]     # (추천)
    for r in rows:
        name = (r.get("menu_name") or "").strip()
        desc = (r.get("description") or "").strip()
        txt = f"{name}. {desc}".strip()
        texts.append(txt)

    # 임베딩
    print(f"Encoding {len(texts)} items ...")
    embs = model.encode(
        texts, batch_size=args.batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )

    if args.dry_run:
        print("Dry run: 첫 1개 임베딩 샘플:", embs[0][:8], "...")
        return

    # 벌크 인덱싱
    def gen_actions():
        for r, e in zip(rows, embs):
            _id = r.get("menu_id") or r.get("id")
            if not _id:
                continue
            src = build_source(r, e)
            yield {
                "_index": args.index,
                "_id": _id,
                "_op_type": "index",
                "_source": src,
            }

    print("Bulk indexing ...")
    ok, fail = helpers.bulk(client, gen_actions(), chunk_size=1000, request_timeout=120, stats_only=True)
    print(f"Bulk done: ok={ok}, failed={fail}")

    # 간단 검증: 문서 수
    try:
        count = client.count(index=args.index).get("count")
        print(f"Indexed docs in {args.index}: {count}")
    except Exception as e:
        print("count 확인 실패:", e)


if __name__ == "__main__":
    main()
