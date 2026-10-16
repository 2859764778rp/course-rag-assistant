import os

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import config


class _EmbeddingsAdapter(Embeddings):
    """把自定义 EmbeddingClient 适配为 LangChain Embeddings 接口。"""

    def __init__(self, client):
        self._client = client

    def embed_documents(self, texts):
        return self._client.embed_documents(texts)

    def embed_query(self, text):
        return self._client.embed_query(text)


class VectorStore:
    def __init__(self, embedding_client, persist_dir: str | None = None):
        self.client = embedding_client
        self.persist_dir = persist_dir or config.CHROMA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self._store = Chroma(
            collection_name="course_kb",
            embedding_function=_EmbeddingsAdapter(self.client),
            persist_directory=self.persist_dir,
        )

    def add_chunks(self, chunks: list[dict], doc_id: str):
        """chunks: [{content, meta}]；doc_id 为关系库中的文档主键。"""
        docs = [
            Document(page_content=c["content"], metadata={**c["meta"], "doc_id": doc_id})
            for c in chunks
        ]
        ids = [f"{doc_id}-{c['meta']['chunk_index']}-{i}" for i, c in enumerate(chunks)]
        self._store.add_documents(docs, ids=ids)
        return len(chunks)

    def delete_doc(self, doc_id: str):
        """按 doc_id 删除该文档全部向量（元数据过滤）。"""
        self._store.delete(where={"doc_id": doc_id})

    def search(self, query: str, top_k: int = None, with_score: bool = True):
        """向量检索。返回 list[Document] 或 (doc, score) 列表。"""
        top_k = top_k or config.RETRIEVER_TOP_K
        if with_score:
            docs = self._store.similarity_search_with_score(query, k=top_k)
            return docs
        return self._store.similarity_search(query, k=top_k)

    def count(self) -> int:
        return self._store._collection.count()


if __name__ == "__main__":
    from embedding import make_client
    emb = make_client()
    vs = VectorStore(emb)
    print("当前向量库 chunk 数:", vs.count())
