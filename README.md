# 近屿智能课程咨询助手

课程咨询助手项目：**阶段二**完成 RAG 问答（模块1）、意图识别（模块2）、文件处理系统（2.3）；**阶段三**完成意图识别正式化（模块3）与网络搜索模块真实化（模块4，接入博查 Web Search API）。第四阶段统一集成到前端（`app.py`）。

## 技术栈

- **LLM**：DeepSeek 云端 API（`deepseek-chat`，OpenAI 兼容接口）
- **Embedding**：本地 `BAAI/bge-large-zh-v1.5`（中文优化，1024 维）
- **向量库**：Chroma（`phase2/data/chroma/`）
- **元数据**：SQLite（`phase2/data/meta.db`）
- **对象存储**：本地目录抽象（接口兼容 S3，可换 MinIO）
- **网络搜索**：博查 Web Search API（`auto` 自动切换：有网真实搜索 / 无网自动降级 mock 兜底）

## 环境准备（3 步）

### 1. 创建虚拟环境并安装依赖

```bash
# Windows PowerShell
cd <项目根目录>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> 若 `torch` 安装过慢，可换 CPU 源：`pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. 配置 API key

在项目根目录创建 `.env` 文件：

```ini
DEEPSEEK_API_KEY=sk-你的密钥
BOCHA_API_KEY=sk-你的博查密钥
```

> 未配置时程序会直接报错提示，不会静默降级。

可选配置：搜索后端（默认 `auto`：有网自动用博查真实搜索，无网自动降级兜底话术，网络恢复后自动切回）：

```ini
SEARCH_BACKEND=auto    # auto 自动切换(默认) / bocha 强制真实 / mock 强制兜底
```

### 3. 构建知识库（首次运行，需联网下载 embedding 模型）

```bash
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内镜像，加速模型下载
python build_kb.py --backend local --model BAAI/bge-large-zh-v1.5
```

构建完成后生成 19 个知识块（示例语料 3 份，位于 `phase2/data/raw/`，可自行替换为课程真实文档）。

## 运行演示

```bash
# RAG 问答演示（对话式）
python demo_rag.py

# 意图识别演示（对话式）
python demo_intent.py

# Embedding 模型对比实验
python eval_embedding.py --backend local --model BAAI/bge-large-zh-v1.5
```

## 代码结构

```
项目根目录/
├── app.py                  # 阶段一：Gradio 聊天前端（阶段四集成三模块）
├── config.py               # 全局配置（LLM/Embedding/阈值/存储路径，唯一改动点）
├── 模块1 RAG 问答
│   ├── preprocess.py       #   语料清洗
│   ├── chunking.py         #   智能分块（标题优先 + Recursive 600/100）
│   ├── embedding.py        #   Embedding 客户端（本地 bge / OpenAI 双后端）
│   ├── vectorstore.py      #   Chroma 封装
│   ├── retriever.py        #   检索器（向量 + BM25 可选混合）
│   └── qa.py               #   QAEngine（LangChain 链，返回答案+来源）
├── 模块2/3 意图识别与搜索
│   ├── categories.py       #   意图类别定义（4 类）
│   ├── router.py           #   分类路由 + 校验 + fallback
│   └── search.py           #   搜索引擎（博查 bocha 真实后端 / mock 可切换）
├── 2.3 文件处理
│   ├── parser.py           #   六格式解析 + PDF 魔数校验
│   └── storage.py          #   对象存储 + SQLite 元数据
├── 脚本
│   ├── build_kb.py         #   构建知识库
│   ├── eval_embedding.py   #   Embedding 对比实验
│   ├── demo_rag.py         #   RAG 问答演示
│   └── demo_intent.py      #   意图识别演示
└── phase2/
    ├── data/               # raw/ 示例语料；chroma/meta.db/objects/corpus 运行时产物
    └── docs/               # 方案设计 + 实验报告
```

## 验证结果摘要

| 模块 | 结果 |
|---|---|
| RAG 问答 | 10 个咨询问题 10/10 通过，回答附真实来源 |
| Embedding 实验 | bge-large-zh：Recall@5 100%、MRR 0.687；bge-small-zh：Recall@5 100%、MRR 0.858 |
| 意图识别 | 4 类样例全部正确，低置信度/异常输入触发 fallback |
| 文件处理 | PDF 校验、六格式解析、对象存储 + SQLite + 向量入库全通 |
| 网络搜索（阶段三） | 博查 Web Search API 接入；auto 自动切换实测：有网真实 / 断网降级 mock / 恢复自动切回 |
| 第四阶段集成 | memory.py 会话/用户 SQLite 持久化、qa 历史注入、router 多轮指代、recommend 追问推荐、app.py 全链路接线；浏览器端到端实测通过 |
| RAG 联网兜底（方案A） | 检索为空或 top1 距离 > 0.95 自动转网络搜索；实测：库内走本地 / 库外与弱相关走联网 / 断网如实提示 |

详细设计见 `phase2/docs/方案.md`，实验细节见 `phase2/docs/实验报告.md`。
