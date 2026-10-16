import argparse
import os
import time

import config
from embedding import make_client
from vectorstore import VectorStore


DEFAULT_EVAL = [
    ("机器学习课程需要什么基础？", ["数学", "编程", "基础"]),
    ("课程有线上直播吗？", ["直播", "回放"]),
    ("报名流程是怎样的？", ["报名", "支付", "助教"]),
    ("课程价格是多少？", ["价格", "优惠"]),
    ("课程可以试听吗？", ["试听"]),
    ("课程学习时长多久？", ["3-6个月", "时长", "周"]),
    ("有没有项目实战？", ["项目", "实战", "证书"]),
    ("哪些课程方向？", ["机器学习", "深度学习", "计算机视觉", "自然语言"]),
]


def load_eval() -> list[tuple[str, list[str]]]:
    path = os.path.join(config.DATA_DIR, "eval_questions.txt")
    if os.path.exists(path):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "||" in line:
                    q, kws = line.split("||", 1)
                    items.append((q.strip(), [k.strip() for k in kws.split(",") if k.strip()]))
        if items:
            return items
    return DEFAULT_EVAL


def run(backend: str | None = None, model: str | None = None, k: int = 5):
    eval_set = load_eval()
    emb = make_client(backend, model)
    print(f"== 模型: {emb.describe()}  评估集: {len(eval_set)} 题  top_k={k} ==")

    tag = emb.model_name.replace("/", "_").replace(".", "_")
    persist_dir = os.path.join(config.CHROMA_DIR, f"eval_{tag}")
    vs = VectorStore(emb, persist_dir=persist_dir)

    import chunking
    corpus_dir = config.CORPUS_DIR
    n_chunks = 0
    t0 = time.time()
    for name in sorted(os.listdir(corpus_dir)):
        p = os.path.join(corpus_dir, name)
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunking.split_text(text, source=name)
        if not chunks:
            continue
        vs.add_chunks(chunks, f"eval-{name}")
        n_chunks += len(chunks)
    build_sec = time.time() - t0
    print(f"入库 {n_chunks} chunks，耗时 {build_sec:.1f}s")

    hits, mrr_sum, total_q = 0, 0.0, 0
    q_times = []
    for q, kws in eval_set:
        t0 = time.time()
        docs = vs.search(q, top_k=k, with_score=False)
        q_times.append(time.time() - t0)
        total_q += 1
        contents = [d.page_content for d in docs]
        hit_rank = None
        for rank, c in enumerate(contents):
            if any(kw in c for kw in kws):
                hit_rank = rank
                break
        if hit_rank is not None:
            hits += 1
            mrr_sum += 1.0 / (hit_rank + 1)
        else:
            print(f"  [MISS] {q}")

    recall = hits / total_q
    mrr = mrr_sum / total_q
    avg_q = sum(q_times) / len(q_times)
    print(f"\n== 结果 ==")
    print(f"Recall@{k}: {recall:.2%} ({hits}/{total_q})")
    print(f"MRR      : {mrr:.4f}")
    print(f"平均查询耗时: {avg_q*1000:.0f} ms | 入库耗时: {build_sec:.1f}s")

    if emb.backend == "openai":
        est_tokens = n_chunks * 800  
        cost = est_tokens / 1_000_000 * 0.02  
        print(f"估算 embedding 成本: ~${cost:.4f}（{est_tokens} tokens）")
    else:
        print("成本: 本地模型 $0（不计电费/算力）")
    return {"model": emb.describe(), "recall@k": recall, "mrr": mrr, "avg_query_ms": avg_q * 1000}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "local"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    run(args.backend, args.model, args.k)
