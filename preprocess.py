import os
import re
import unicodedata

import fitz  

import config



def clean_text(text: str) -> str:
    """通用文本清洗：去页眉页脚/页码、归一化空白、去控制字符。"""
    if not text:
        return ""
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
   
    text = unicodedata.normalize("NFKC", text)
  
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if re.fullmatch(r"[\d\s\-—|·]+", s):          
            continue
        if re.fullmatch(r"第\s*\d+\s*页?", s):        
            continue
        if re.search(r"copyright|版权所有|©", s, re.I) and len(s) < 60:
            continue
        lines.append(s)
    text = "\n".join(lines)
   
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




def _read_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


def _read_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def _read_plain(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


_READERS = {".txt": _read_plain, ".md": _read_plain, ".pdf": _read_pdf, ".html": _read_html}


def extract_text(path: str) -> str:
    """按扩展名提取原始文本（未清洗）。docx/pptx 抛 NotImplementedError 提示走 fileproc。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".docx", ".pptx", ".ipynb"):
        raise NotImplementedError(f"{ext} 请先通过 phase2.modules.fileproc.parser 解析为文本")
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"不支持的语料格式: {ext}")
    return reader(path)


def process_file(src_path: str, out_dir: str | None = None) -> str:
    """清洗单个文件：读取 → 清洗 → 写入 corpus，返回输出路径。"""
    out_dir = out_dir or config.CORPUS_DIR
    os.makedirs(out_dir, exist_ok=True)
    raw = extract_text(src_path)
    cleaned = clean_text(raw)
    name = os.path.splitext(os.path.basename(src_path))[0] + ".txt"
    out_path = os.path.join(out_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return out_path


def process_dir(raw_dir: str | None = None, out_dir: str | None = None) -> list[str]:
    """批量清洗 raw 目录下全部支持的文件。"""
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.CORPUS_DIR
    results = []
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in _READERS:
            continue
        try:
            out = process_file(path, out_dir)
            results.append(out)
        except Exception as e: 
            print(f"[warn] 清洗失败 {name}: {e}")
    return results


if __name__ == "__main__":
    outs = process_dir()
    print(f"清洗完成 {len(outs)} 个文件:")
    for o in outs:
        print(" -", o)
