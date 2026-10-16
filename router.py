import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config
from categories import build_category_desc, is_valid
from search import make_search_engine
from dataclasses import dataclass, field


@dataclass
class IntentResult:
    intent: str
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""
    fallback_used: bool = False
    fallback_info: str = ""


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是意图识别模型。请从以下类别中选择用户输入的意图：\n"
        "{categories}\n"
        "规则：\n"
        "1. 只能返回一个类别（key 形式，如 course_qa）；\n"
        "2. 必须只输出 JSON，不要输出任何其他文字：\n"
        '   {{"intent": "<类别key>", "confidence": 0~1之间的数字, "reason": "<一句话理由>"}}\n'
        "3. confidence 表示你对分类的把握程度，0 最低、1 最高。\n"
        "4. 结合「最近对话上下文」理解当前输入中的指代（如「它/这个/那门课」）。\n"
        "5. 重要：chitchat（闲聊）只用于纯打招呼、寒暄、自我介绍、感谢（如「你好」「你是谁」「谢谢」）；"
        "凡询问具体数据/数量/最新/今天/新闻/时事/发展趋势/原因的事实性问题（如「2026年新生儿数量」「AI最新进展」），"
        "一律归 pro_consult 或 course_qa，绝不能判为 chitchat。\n"
        "6. 重要：file_ops 只用于对文件本身的管理动作（上传/删除/列出文件/确认是否传成功）；"
        "凡是问「文档内容是什么/总结文档/文档里讲了什么/根据上传的文档回答某问题」，都是在提问文档内容，"
        "必须归 course_qa 或 pro_consult，不能判 file_ops。\n"
    )),
    ("human", "最近对话上下文：{recent_context}\n当前用户输入：{input}"),
])


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON（容忍 ```json 围栏与前后杂文本）。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class IntentRouter:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.0,          
            timeout=config.INTENT_LLM_TIMEOUT,
        )
        self._chain = INTENT_PROMPT | self.llm
        self.search = make_search_engine()

    def classify(self, user_input: str, recent_context: str = "") -> IntentResult:
        """核心：LLM 分类 → 校验 → fallback。
        recent_context: 最近对话用户输入摘要（多轮指代，如「它呢？」需上下文判断意图）。"""
        result = IntentResult(intent="unknown", raw="")
        try:
            resp = self._chain.invoke({
                "categories": build_category_desc(),
                "input": user_input,
                "recent_context": recent_context or "（无）",
            })
            raw = resp.content if hasattr(resp, "content") else str(resp)
            result.raw = raw
            data = _extract_json(raw)
            if not data:
                result.reason = "LLM 输出无法解析为 JSON"
                return self._fallback(user_input, result)

            intent = str(data.get("intent", "")).strip().lower()
            conf = float(data.get("confidence", 0) or 0)
            result.intent = intent
            result.confidence = conf
            result.reason = str(data.get("reason", ""))

            if not is_valid(intent):
                result.reason = f"意图类别非法: {intent}"
                return self._fallback(user_input, result)
            if conf < config.INTENT_CONFIDENCE_THRESHOLD:
                result.reason = f"置信度不足: {conf:.2f} < {config.INTENT_CONFIDENCE_THRESHOLD}"
                return self._fallback(user_input, result)
            return result
        except Exception as e:  
            result.reason = f"LLM 调用异常: {type(e).__name__}: {e}"
            return self._fallback(user_input, result)

    def _fallback(self, user_input: str, result: IntentResult) -> IntentResult:
        """Fallback 补救链：网络搜索 → 兜底话术。"""
        result.fallback_used = True
        try:
            sr = self.search.search(user_input)
            if sr.snippets:
                result.fallback_info = "；".join(sr.snippets[:2])
                return result
        except Exception as e:  
            result.fallback_info = f"搜索失败: {e}"
        result.fallback_info = "抱歉，我暂时无法准确判断您的需求，建议直接咨询课程助教。"
        return result


if __name__ == "__main__":
    router = IntentRouter()
    for q in ["机器学习课程多少钱？", "你好呀", "帮我看看上传的文件", "什么是大模型微调？"]:
        r = router.classify(q)
        print(f"{q!r:24} -> {r.intent:12} conf={r.confidence:.2f} fallback={r.fallback_used}")
