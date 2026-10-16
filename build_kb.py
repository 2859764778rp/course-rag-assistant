import argparse
import os

import config
import preprocess, chunking
from embedding import make_client
from vectorstore import VectorStore
from storage import MetadataDB


def build(backend: str | None = None, model: str | None = None, reset: bool = False):
    print("==> [1/4] 清洗 raw 目录语料")
    corpus_files = preprocess.process_dir()
    if not corpus_files:
        print("!! raw 目录没有可清洗的语料文件，请先放入课程文档到", config.RAW_DIR)
        return

    print("==> [2/4] 初始化 embedding")
    emb = make_client(backend, model)
    print("    使用:", emb.describe())

    print("==> [3/4] 分块 + 向量化入库")
    vs = VectorStore(emb)
    db = MetadataDB()
    if reset:
        print("    [reset] 清空旧向量库...")
        try:
            vs._store.delete_collection()
        except Exception as e:  
            print(f"    [warn] delete_collection: {e}")
        vs = VectorStore(emb)

    total = 0
    for cf in corpus_files:
        with open(cf, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunking.split_text(text, source=os.path.basename(cf))
        if not chunks:
            continue
        doc_id = db.create_document(os.path.basename(cf), "txt", len(text.encode("utf-8")), "corpus:" + cf)
        db.add_chunks(doc_id, chunks)
        vs.add_chunks(chunks, doc_id)
        db.update_status(doc_id, "vectorized")
        total += len(chunks)
        print(f"    + {os.path.basename(cf)}: {len(chunks)} chunks")

    print(f"==> [4/4] 完成，共入库 {total} 个 chunk（向量库总数 {vs.count()}）")
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "local"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reset", action="store_true", help="重建前清空向量库")
    args = ap.parse_args()
    build(args.backend, args.model, args.reset)
