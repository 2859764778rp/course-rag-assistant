import os

import numpy as np

import config


class EmbeddingClient:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or config.EMB_BACKEND
        self.model_name = model or self._default_model()
        self._client = None
        self._init_backend()

    def _default_model(self) -> str:
        if self.backend == "openai":
            return config.EMB_OPENAI_MODEL
        return config.EMB_LOCAL_MODEL

    def _init_backend(self):
        if self.backend == "openai":
            if not config.EMB_OPENAI_KEY:
                raise RuntimeError("缺少 OPENAI_API_KEY 环境变量（embedding 用 OpenAI 后端时必填）")
            from openai import OpenAI
            self._client = OpenAI(api_key=config.EMB_OPENAI_KEY, base_url=config.EMB_OPENAI_BASE_URL)
        elif self.backend == "local":
           
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer
            self._model = None
            self._st_class = SentenceTransformer
        else:
            raise ValueError(f"未知 embedding 后端: {self.backend}")

    def _ensure_local_model(self):
        if self._model is None:
           
            try:
                self._model = self._st_class(self.model_name, local_files_only=True)
                return
            except Exception:
                pass 
            try:
                self._model = self._st_class(self.model_name)
            except Exception as e:
                print(f"[warn] 加载 {self.model_name} 失败({e})，降级为 {config.EMB_LOCAL_FALLBACK}")
                self.model_name = config.EMB_LOCAL_FALLBACK
                self._model = self._st_class(self.model_name, local_files_only=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "openai":
            resp = self._client.embeddings.create(model=self.model_name, input=texts)
            return [d.embedding for d in resp.data]
        self._ensure_local_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def describe(self) -> str:
        return f"{self.backend}:{self.model_name}"


def make_client(backend: str | None = None, model: str | None = None) -> EmbeddingClient:
    """工厂：构造失败时按 openai→local(bge) 顺序降级尝试。"""
    order = [backend] if backend else [config.EMB_BACKEND]
    if order[0] != "openai":
        order.append("openai")
    if "local" not in order:
        order.append("local")
    last_err = None
    for b in order:
        if b is None:
            continue
        try:
            return EmbeddingClient(b, model)
        except Exception as e:  
            last_err = e
            print(f"[warn] embedding 后端 {b} 不可用: {e}")
    raise RuntimeError(f"所有 embedding 后端均不可用: {last_err}")


if __name__ == "__main__":
    c = make_client()
    print("使用:", c.describe())
    v = c.embed_query("近屿智能开设了哪些人工智能课程？")
    print("维度:", len(v))
