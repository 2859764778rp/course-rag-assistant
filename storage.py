import os
import sqlite3
import time
import uuid

import config



class LocalObjectStorage:
    """本地目录版对象存储：语义对齐 S3 put/get/delete/exists。"""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or config.LOCAL_STORE_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
    
        key = key.replace("\\", "/").lstrip("/")
        return os.path.normpath(os.path.join(self.base_dir, key))

    def put(self, src_path: str, key: str | None = None) -> str:
        import shutil
        key = key or f"{uuid.uuid4().hex}/{os.path.basename(src_path)}"
        dst = self._path(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src_path, dst)
        return key

    def get(self, key: str, dst_path: str) -> str:
        import shutil
        shutil.copy2(self._path(key), dst_path)
        return dst_path

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if os.path.exists(p):
            os.remove(p)
            return True
        return False

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def url(self, key: str) -> str:
        return os.path.join(self.base_dir, key)


class MinioObjectStorage:
    """MinIO/S3 版对象存储（预留；配置 MINIO_* 后可用）。"""

    def __init__(self, endpoint=None, access=None, secret=None, bucket=None):
        from minio import Minio
        endpoint = endpoint or config.MINIO_ENDPOINT
        access = access or config.MINIO_ACCESS_KEY
        secret = secret or config.MINIO_SECRET_KEY
        bucket = bucket or config.MINIO_BUCKET
        self.client = Minio(endpoint, access_key=access, secret_key=secret, secure=False)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put(self, src_path: str, key: str | None = None) -> str:
        key = key or f"{uuid.uuid4().hex}/{os.path.basename(src_path)}"
        self.client.fput_object(self.bucket, key, src_path)
        return key

    def get(self, key: str, dst_path: str) -> str:
        self.client.fget_object(self.bucket, key, dst_path)
        return dst_path

    def delete(self, key: str) -> bool:
        try:
            self.client.remove_object(self.bucket, key)
            return True
        except Exception:  
            return False

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception: 
            return False

    def url(self, key: str) -> str:
        return f"{self.bucket}/{key}"


def make_object_storage():
    """工厂：按 config.OBJ_STORE_BACKEND 返回实现；未知/异常回退本地。"""
    backend = config.OBJ_STORE_BACKEND
    if backend == "minio":
        try:
            return MinioObjectStorage()
        except Exception as e:  
            print(f"[warn] MinIO 不可用({e})，回退本地存储")
    return LocalObjectStorage()



_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    object_key TEXT,
    status TEXT NOT NULL DEFAULT 'uploaded',   -- uploaded|parsing|vectorized|failed
    ext TEXT,
    size INTEGER,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    chunk_index INTEGER,
    content TEXT,
    vector_id TEXT,
    chapter TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class MetadataDB:
    """SQLite 元数据操作。"""

    def __init__(self, path: str | None = None):
        self.path = path or config.SQLITE_PATH
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def create_document(self, filename: str, ext: str, size: int, object_key: str) -> str:
        doc_id = uuid.uuid4().hex[:16]
        self.conn.execute(
            "INSERT INTO documents(id, filename, object_key, status, ext, size, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, filename, object_key, "uploaded", ext, size, time.time()),
        )
        self.conn.commit()
        return doc_id

    def update_status(self, doc_id: str, status: str):
        self.conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))
        self.conn.commit()

    def add_chunks(self, doc_id: str, chunks: list[dict]):
        rows = [
            (doc_id, c["meta"].get("chunk_index", i), c["content"],
             f"{doc_id}-{i}", c["meta"].get("chapter", ""))
            for i, c in enumerate(chunks)
        ]
        self.conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_index, content, vector_id, chapter) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def get_document(self, doc_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM documents").description]
        return dict(zip(cols, row))

    def list_documents(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM documents").description]
        return [dict(zip(cols, r)) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0 or True

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = MetadataDB()
    print("SQLite 就绪:", db.path)
    print("现有文档:", db.list_documents())
    db.close()
