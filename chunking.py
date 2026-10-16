import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


_HEADING_RE = re.compile(
    r"^\s*(#{1,6}\s+|第[0-9一二三四五六七八九十百]+[章篇节]\s*|"
    r"[一二三四五六七八九十]+、|\d+(\.\d+)*[\s、.．])",
)


def _split_by_headings(text: str):
    """按章节标题把文本切成 (chapter_title, body) 段。"""
    lines = text.split("\n")
    sections = []
    cur_title, cur_lines = "前言", []
    for ln in lines:
        s = ln.strip()
        if _HEADING_RE.match(s) and len(s) <= 60:
            if cur_lines:
                sections.append((cur_title, "\n".join(cur_lines).strip()))
            cur_title = s
            cur_lines = [s]
        else:
            cur_lines.append(ln)
    if cur_lines:
        sections.append((cur_title, "\n".join(cur_lines).strip()))
    return [(t, b) for t, b in sections if b]


def split_text(text: str, source: str, chunk_size: int = None, overlap: int = None):
    """把一篇清洗后的文档切成 chunk 列表。"""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = []
    for chapter, body in _split_by_headings(text):
        if len(body) <= chunk_size * 1.2:
            pieces = [body]
        else:
            pieces = splitter.split_text(body)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "content": piece,
                "meta": {
                    "source": os.path.basename(source),
                    "chapter": chapter,
                    "chunk_index": i,
                },
            })
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.CORPUS_DIR, "course_intro.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text, path)
    print(f"共 {len(chunks)} 个 chunk")
    for c in chunks[:3]:
        print("---", c["meta"], "---")
        print(c["content"][:120])
