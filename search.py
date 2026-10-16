import time
from dataclasses import dataclass, field

import requests

import config


@dataclass
class SearchResult:
    query: str
    snippets: list[str] = field(default_factory=list)
    source: str = "mock"


class MockSearchEngine:
    """兜底 mock：不依赖网络，返回引导性摘要，用于断网降级与演示。"""

    def search(self, query: str, top_k: int = 3) -> SearchResult:
        return SearchResult(
            query=query,
            snippets=[
                "（当前为 mock 结果）网络搜索不可用或未配置，暂时无法获取实时信息。",
                "建议您补充更多上下文，或直接咨询课程助教获取人工解答。",
            ],
        )


_NETWORK_ERR_NAMES = (
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "ProxyError",
    "SSLError",
    "ConnectionResetError",
    "ChunkedEncodingError",
    "RemoteDisconnected",
)


class BochaSearchEngine:
    """模块4：博查 Web Search API 真实网络搜索。

    - 端点：POST https://api.bochaai.com/v1/web-search
    - 鉴权：Authorization: Bearer <BOCHA_API_KEY>
    - 错误码：401 key 无效 / 403 余额不足 / 429 频率限制 / 500 服务端
    - 任何异常都返回带原因的兜底 snippets，保证上游（router）链路不崩溃。
    """

    def __init__(self, api_key: str | None = None, timeout: int | None = None):
        self.api_key = api_key or config.BOCHA_API_KEY
        self.url = config.BOCHA_SEARCH_URL
        self.timeout = timeout or config.BOCHA_SEARCH_TIMEOUT
        self.count = max(1, min(config.BOCHA_SEARCH_COUNT, 50))

    def search(self, query: str, top_k: int = 3) -> SearchResult:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "query": query,
                "summary": True,
                "freshness": "noLimit",
                "count": self.count,
            }
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return self._fail(query, f"HTTP {resp.status_code}")

            j = resp.json()
            if j.get("code") != 200:
                return self._fail(query, f"code={j.get('code')} msg={j.get('msg')}")

            pages = ((j.get("data") or {}).get("webPages") or {}).get("value") or []
            snippets = []
            for p in pages[:top_k]:
                name = (p.get("name") or "").strip()
                url = (p.get("url") or "").strip()
                summary = (p.get("summary") or p.get("snippet") or "").strip()
                parts = [x for x in (name, summary, url) if x]
                if parts:
                    snippets.append(" | ".join(parts))
            if not snippets:
          
                return SearchResult(
                    query=query,
                    snippets=[f"未检索到与「{query}」相关的网络结果。"],
                    source="bocha",
                )
            return SearchResult(query=query, snippets=snippets, source="bocha")
        except Exception as e:  
            return self._fail(query, f"{type(e).__name__}: {e}")

    def _fail(self, query: str, reason: str) -> SearchResult:
        return SearchResult(
            query=query,
            snippets=[
                f"（博查搜索失败：{reason}）暂时无法获取实时信息。",
                "建议您稍后重试，或直接咨询课程助教。",
            ],
            source="bocha",
        )


class AutoSearchEngine:
    """auto 后端：有网用博查真实搜索，没网/接口不可用自动降级 mock，网络恢复后自动切回。

    机制：
    - 每次调用先尝试博查（短超时）；成功 → 返回真实结果
    - 网络类失败（连不上/超时/5xx/429）→ 进入冷却期，冷却期内直接用 mock 兜底话术
    - 冷却结束后自动重试博查，网络恢复即自动恢复真实搜索
    - 未配置 BOCHA_API_KEY 时直接 mock（不抛错）
    """

    def __init__(self, cooldown: int | None = None, timeout: int | None = None):
        self.cooldown = cooldown if cooldown is not None else getattr(config, "AUTO_SEARCH_COOLDOWN", 60)
        self._timeout = timeout if timeout is not None else getattr(config, "AUTO_SEARCH_TIMEOUT", 8)
        self._bocha = BochaSearchEngine(timeout=self._timeout)
        self._mock = MockSearchEngine()
        self._degraded_until = 0.0

    def search(self, query: str, top_k: int = 3) -> SearchResult:
        if not config.BOCHA_API_KEY:
            return self._mock_result(query, top_k, "未配置 BOCHA_API_KEY")

        now = time.time()
        if now < self._degraded_until:
            wait = int(self._degraded_until - now)
            return self._mock_result(query, top_k, f"网络不可用（冷却中，约 {wait}s 后自动重试）")

        try:
            r = self._bocha.search(query, top_k)
        except Exception as e:  
            self._degraded_until = now + self.cooldown
            return self._mock_result(query, top_k, f"{type(e).__name__}: {e}")

        if r.source == "bocha" and not (r.snippets and r.snippets[0].startswith("（博查搜索失败")):
            return r 

        reason = r.snippets[0] if r.snippets else "未知错误"
        if self._is_transient_fail(reason):
           
            self._degraded_until = now + self.cooldown
            return self._mock_result(query, top_k, reason)
        return r 

    @staticmethod
    def _is_transient_fail(reason: str) -> bool:
        if any(n in reason for n in _NETWORK_ERR_NAMES):
            return True
        return "HTTP 5" in reason or "HTTP 429" in reason or "code=429" in reason or "code=500" in reason

    def _mock_result(self, query: str, top_k: int, why: str) -> SearchResult:
        r = self._mock.search(query, top_k)
        r.source = "mock(auto)"
        r.snippets.insert(0, f"（网络搜索不可用：{why[:80]}，已自动切换兜底模式）")
        return r


class SearchEngine:
    """搜索入口：按配置切换 auto（默认）/ bocha / mock。"""

    def __init__(self, backend: str | None = None):
        self.backend = backend or getattr(config, "SEARCH_BACKEND", "auto")
        if self.backend == "mock":
            self._impl = MockSearchEngine()
        elif self.backend == "bocha":
            self._impl = BochaSearchEngine()
        elif self.backend == "auto":
            self._impl = AutoSearchEngine()
        else:
            raise NotImplementedError(f"未知搜索后端: {self.backend}")

    def search(self, query: str, top_k: int = 3) -> SearchResult:
        return self._impl.search(query, top_k=top_k)


def make_search_engine() -> SearchEngine:
    """工厂：默认按 config.SEARCH_BACKEND 创建；未知后端时回退 mock。"""
    backend = getattr(config, "SEARCH_BACKEND", "auto")
    try:
        return SearchEngine(backend)
    except (NotImplementedError, RuntimeError):
        return SearchEngine("mock")


if __name__ == "__main__":
    eng = make_search_engine()
    r = eng.search("人工智能课程学费")
    print(r.source, r.snippets[:2])
