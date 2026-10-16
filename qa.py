from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from retriever import Retriever

SYSTEM_PROMPT = (
    "你是一位课程助教，请根据以下已知信息用简明中文回答用户问题。\n"
    "要求：\n"
    "1. 优先依据已知信息回答；结合之前的对话历史理解用户指代（如「它/这个/那门课」）；已知信息不完整时，先给出已知的部分；\n"
    "2. 只有已知信息与问题完全无关时，才回答「资料中暂未找到相关内容」；\n"
    "3. 不要自行输出「来源」行，来源信息由系统自动附加。\n\n"
    "已知信息：\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
])

_CONTEXT_SEP = "\n\n---\n\n"


def _format_sources(docs) -> str:
    seen, parts = set(), []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "未知")
        chap = meta.get("chapter", "")
        key = f"{src}|{chap}"
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{src}（{chap}）")
    return "；".join(parts[:3]) or "无"


class QAEngine:
    """RAG 问答引擎：本地检索 + 生成；检索不可靠时（方案A）自动转网络搜索兜底。
    第四阶段：ask() 接收外部对话历史（SessionStore 持久化），实现上下文记忆。
    """

    def __init__(self, retriever: Retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
        )
        self._history: dict[str, BaseChatMessageHistory] = {}
        self._search = None

   

    def _need_web_fallback(self, question: str) -> tuple[bool, list, list]:
        """返回 (是否需网络兜底, docs, scores)。"""
        hits = self.retriever.retrieve(question, top_k=config.RETRIEVER_TOP_K, with_score=True)
        docs = [d for d, _ in hits]
        scores = [s for _, s in hits]
        if not config.RAG_SEARCH_FALLBACK:
            return False, docs, scores
        if not docs:
            return True, docs, scores
        if scores[0] > config.RAG_FALLBACK_DISTANCE:
            return True, docs, scores
        return False, docs, scores

    def _get_search(self):
        if self._search is None:
            from search import make_search_engine
            self._search = make_search_engine()
        return self._search

    

    def _invoke_llm(self, question: str, context: str, history: list | None = None) -> str:
        """messages = system(带已知信息) + 最近N轮历史 + 当前问题。"""
        messages = QA_PROMPT.format_messages(context=context)
        if history:
            window = config.HISTORY_WINDOW * 2
            for m in history[-window:]:
                if m.get("role") == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m.get("role") == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=question))
        resp = self.llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def _append_source(self, answer: str, sources: str) -> str:
        if sources and sources != "无" and "来源：" not in answer:
            return answer + f"\n\n来源：{sources}"
        return answer

    def _ask_local(self, question: str, docs: list, history: list | None = None) -> tuple[str, str]:
        context = _CONTEXT_SEP.join(d.page_content for d in docs)
        answer = self._invoke_llm(question, context, history)
        sources = _format_sources(docs)
        return self._append_source(answer, sources), sources

    def _ask_via_search(self, question: str, history: list | None = None) -> tuple[str, str, object]:
        """网络搜索兜底：返回 (answer, sources, SearchResult)。"""
        sres = self._get_search().search(question)

        if sres.source != "bocha":
            note = sres.snippets[0] if sres.snippets else "未知原因"
            msg = (
                f"本地资料库未检索到与「{question}」直接相关的内容，"
                f"且网络搜索当前不可用（{note}）。建议补充更多上下文，或直接咨询课程助教。"
            )
            return msg, "无", sres

        if sres.snippets and sres.snippets[0].startswith("未检索到与"):
            return f"本地资料库与网络搜索均未找到与「{question}」直接相关的内容。", "无", sres

        context = _CONTEXT_SEP.join(sres.snippets)
        answer = self._invoke_llm(question, context, history)
        return self._append_source(answer, "网络搜索"), "网络搜索", sres


    def ask(self, question: str, history: list | None = None, return_docs: bool = False):
        """提问 → (answer, sources)。
        history: 外部传入的对话历史 [{"role","content"}, ...]（SessionStore 持久化管理）。
        return_docs=True 时返回检索文档/搜索结果。"""
        need_search, docs, scores = self._need_web_fallback(question)
        if need_search:
            answer, sources, sres = self._ask_via_search(question, history)
            if return_docs:
                return answer, sources, sres
            return answer, sources
        answer, sources = self._ask_local(question, docs, history)
        if return_docs:
            return answer, sources, docs
        return answer, sources

    def chat_direct(self, question: str, history: list | None = None) -> str:
        """无 RAG 的直接对话（闲聊/兜底引导用）：带历史，纯 LLM 回答。"""
        sys_prompt = (
            "你是近屿智能课程咨询助手，用简短友好的中文与学生闲聊，"
            "遇到课程问题时引导学生直接提问（如「请问您想了解哪门课程？」）。"
        )
        messages = [ChatPromptTemplate.from_messages([("system", sys_prompt)]).format_messages()[0]]
        if history:
            window = config.HISTORY_WINDOW * 2
            for m in history[-window:]:
                if m.get("role") == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m.get("role") == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=question))
        resp = self.llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def reset(self, session_id: str = "default"):
        self._history.pop(session_id, None)


class _InMemoryHistory(BaseChatMessageHistory):
    """极简内存历史（阶段二单会话够用，阶段四已由 SessionStore 接管持久化）。"""

    def __init__(self):
        self.messages: list = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []


if __name__ == "__main__":
    from embedding import make_client
    from vectorstore import VectorStore

    emb = make_client()
    vs = VectorStore(emb)
    retriever = Retriever(vs)
    qa = QAEngine(retriever)
    ans, src = qa.ask("机器学习课程需要什么基础？")
    print("回答:", ans)
    print("来源:", src)
