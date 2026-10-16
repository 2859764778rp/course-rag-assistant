from rank_bm25 import BM25Okapi

import config
import vectorstore as vs_mod


class Retriever:
    def __init__(self, vector_store: vs_mod.VectorStore):
        self.vs = vector_store
        self._bm25 = None
        self._bm25_docs = []
        if config.HYBRID_ENABLE:
            self._build_bm25()

    def _build_bm25(self):
        """用向量库现有 chunk 构建 BM25 索引（阶段二语料小，内存构建可接受）。"""
        all_docs = self.vs._store._collection.get(include=["documents", "metadatas"])
        self._bm25_docs = all_docs["documents"] or []
        if self._bm25_docs:
            self._bm25 = BM25Okapi([d.split() for d in self._bm25_docs])

    def retrieve(self, query: str, top_k: int = None, with_score: bool = False):
        top_k = top_k or config.RETRIEVER_TOP_K
        if not config.HYBRID_ENABLE or self._bm25 is None:
            res = self.vs.search(query, top_k=top_k, with_score=True)
            return res if with_score else [d for d, _ in res]


        vec_hits = self.vs.search(query, top_k=top_k * 2, with_score=True)
        bm25_scores = self._bm25.get_scores(query.split())
        order = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[: top_k * 2]

        rrf = {}
        for rank, (doc, _) in enumerate(vec_hits):
            key = doc.page_content
            rrf[key] = rrf.get(key, 0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(order):
            key = self._bm25_docs[idx]
            rrf[key] = rrf.get(key, 0) + 1.0 / (60 + rank + 1)

        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        docs = []
        for content, _ in ranked:
            for d, s in vec_hits:
                if d.page_content == content:
                    docs.append(d)
                    break
        return docs


if __name__ == "__main__":
    from embedding import make_client
    emb = make_client()
    vs = vs_mod.VectorStore(emb)
    r = Retriever(vs)
    docs = r.retrieve("机器学习课程需要什么基础？")
    print(f"检索到 {len(docs)} 条")
    for d in docs[:3]:
        print("---", d.metadata, "---")
        print(d.page_content[:100])
