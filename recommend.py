from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是课程咨询助手。根据刚刚的问答，给出 {n} 个学生最可能继续追问的简短问题。\n"
        "要求：\n"
        "1. 每个问题一行，不要编号、不要序号符号、不要任何解释；\n"
        "2. 问题要具体、口语化，与当前课程咨询话题相关；\n"
        "3. 不要重复用户刚问过的问题。\n\n"
        "学生刚问：{question}\n"
        "你的回答：{answer}"
    )),
    ("human", "请直接输出 {n} 个问题："),
])


class QuestionRecommender:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.7,
            timeout=config.LLM_TIMEOUT,
        )

    def recommend(self, question: str, answer: str, n: int | None = None) -> list[str]:
        if not config.QUESTION_RECOMMEND:
            return []
        n = n or config.RECOMMEND_COUNT
        try:
            resp = self.llm.invoke(
                _PROMPT.format(question=question, answer=(answer or "")[:600], n=n)
            )
            text = resp.content if hasattr(resp, "content") else str(resp)
            out = []
            for line in str(text).splitlines():
                q = line.strip().strip("0123456789.-、)） ")
                if q and len(q) <= 40 and q != question:
                    out.append(q)
                if len(out) >= n:
                    break
            return out
        except Exception:
            return []


if __name__ == "__main__":
    rec = QuestionRecommender()
    print(rec.recommend("机器学习课程多少钱？", "机器学习课程费用约 3000 元，支持分期。"))
