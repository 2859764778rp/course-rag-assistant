from __future__ import annotations

INTENT_CATEGORIES: dict[str, dict] = {
    "course_qa": {
        "label": "课程问答",
        "desc": "询问课程内容、学费、时长、报名方式、试听、师资、证书等课程相关问题",
        "examples": ["有什么课程？", "学费多少？", "怎么报名？", "课程有试听吗？"],
    },
    "file_ops": {
        "label": "文件操作",
        "desc": "对文件本身做管理动作：上传、重新解析、查看已传文件列表、删除文件、询问文件是否传成功。注意：只要是在「提问文件里写了什么/文档内容是什么/总结文档/根据文档回答」，一律不算 file_ops，应归为 course_qa 或 pro_consult（走 RAG 检索文档内容作答）",
        "examples": ["帮我解析这个PDF", "文件传好了吗", "查看我上传的文件", "把刚才的文件删掉"],
    },
    "chitchat": {
        "label": "闲聊",
        "desc": "打招呼、寒暄、自我介绍、表达感谢等非业务性对话",
        "examples": ["你好", "你是谁", "谢谢"],
    },
    "pro_consult": {
        "label": "专业咨询",
        "desc": "询问AI技术概念、就业前景、行业趋势等专业知识问题",
        "examples": ["什么是RAG？", "AI就业前景怎么样？", "大模型和机器学习什么关系"],
    },
}

VALID_INTENTS = set(INTENT_CATEGORIES.keys())


def build_category_desc() -> str:
    """把类别定义拼成提示词中的分类范围文本。"""
    lines = []
    for key, info in INTENT_CATEGORIES.items():
        lines.append(f"- {key}（{info['label']}）：{info['desc']}。示例：{'；'.join(info['examples'])}")
    return "\n".join(lines)


def is_valid(intent: str) -> bool:
    return intent in VALID_INTENTS
