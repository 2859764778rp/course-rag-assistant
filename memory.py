import os
import sqlite3
import time
import uuid

import config


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);
"""


def _now() -> float:
    return time.time()


class SessionStore:
    """会话与用户的持久化存储。"""

    def __init__(self, path: str | None = None):
        self.path = path or config.SQLITE_PATH
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    

    def get_or_create_user(self, nickname: str, uid: str | None = None) -> dict:
        """获取或创建用户。
        uid 固定时以 uid 为身份键（刷新/重启不影响身份）；
        否则按昵称匹配（旧行为，仅兼容）。"""
        nickname = (nickname or "同学").strip() or "同学"
        if uid:
            row = self.conn.execute("SELECT id, nickname FROM users WHERE id=?", (uid,)).fetchone()
            if row:
                return {"id": row[0], "nickname": row[1]}
            self.conn.execute(
                "INSERT INTO users(id, nickname, created_at) VALUES (?,?,?)",
                (uid, nickname, _now()),
            )
            self.conn.commit()
            return {"id": uid, "nickname": nickname}
        row = self.conn.execute("SELECT id, nickname FROM users WHERE nickname=?", (nickname,)).fetchone()
        if row:
            return {"id": row[0], "nickname": row[1]}
        new_uid = "u_" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO users(id, nickname, created_at) VALUES (?,?,?)",
            (new_uid, nickname, _now()),
        )
        self.conn.commit()
        return {"id": new_uid, "nickname": nickname}

    def rename_user(self, user_id: str, nickname: str):
        self.conn.execute("UPDATE users SET nickname=? WHERE id=?", (nickname, user_id))
        self.conn.commit()

    

    def create_session(self, user_id: str, title: str | None = None) -> str:
        sid = "c_" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO sessions(id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (sid, user_id, title or "新对话", _now(), _now()),
        )
        self.conn.commit()
        return sid

    def touch_session(self, chat_id: str, title: str | None = None):
        if title:
            self.conn.execute(
                "UPDATE sessions SET updated_at=?, title=? WHERE id=? AND title='新对话'",
                (_now(), title[:30], chat_id),
            )
        else:
            self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), chat_id))
        self.conn.commit()

    def list_sessions(self, user_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [
            {"chat_id": r[0], "title": r[1] or "新对话", "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def delete_session(self, chat_id: str):
        """删除会话及其全部消息（不可恢复）。"""
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (chat_id,))
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (chat_id,))

  

    def add_message(self, chat_id: str, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?,?,?,?)",
            (chat_id, role, content, _now()),
        )
        self.conn.commit()
        self.touch_session(chat_id)

    def get_history(self, chat_id: str, window: int | None = None) -> list[dict]:
        """返回最近 window 轮（user+assistant）消息，按时间正序。"""
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY ts DESC",
            (chat_id,),
        ).fetchall()
        msgs = [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        if window:
            msgs = msgs[-window * 2:]
        return msgs

    def history_for_router(self, chat_id: str, turns: int | None = None) -> str:
        """给意图识别用的多轮上下文（只取用户侧最近输入）。"""
        turns = turns or config.ROUTER_CONTEXT_TURNS
        msgs = self.get_history(chat_id, window=turns)
        if not msgs:
            return ""
        lines = [f"用户：{m['content']}" for m in msgs if m["role"] == "user"]
        return "；".join(lines[-turns:])

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    store = SessionStore()
    u = store.get_or_create_user("测试同学")
    print("用户:", u)
    cid = store.create_session(u["id"], "首轮测试")
    store.add_message(cid, "user", "机器学习课多少钱？")
    store.add_message(cid, "assistant", "约 3000 元。")
    store.add_message(cid, "user", "它有试听吗？")
    print("历史:", store.get_history(cid, window=5))
    print("路由上下文:", store.history_for_router(cid))
    print("会话列表:", store.list_sessions(u["id"]))
