import sys

from embedding import make_client
from vectorstore import VectorStore
from retriever import Retriever
from qa import QAEngine


def main():
    print("初始化 RAG 引擎...")
    emb = make_client()
    vs = VectorStore(emb)
    print(f"   embedding: {emb.describe()} | 向量库 chunk 数: {vs.count()}")
    if vs.count() == 0:
        print("!! 向量库为空，请先运行: python build_kb.py")
        return
    retriever = Retriever(vs)
    qa = QAEngine(retriever)

    print("\n== RAG 问答演示（输入 q 退出）==\n")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "退出"):
            break
        try:
            ans, src = qa.ask(q)
            print("\n助手:", ans)
            print(f"来源: {src}\n")
        except Exception as e:  
            print(f"[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
