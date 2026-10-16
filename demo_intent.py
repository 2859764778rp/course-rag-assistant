import sys

from router import IntentRouter


def main():
    print("初始化意图识别路由...")
    router = IntentRouter()
    print("\n== 意图识别演示（输入 q 退出）==\n")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "退出"):
            break
        r = router.classify(q)
        print(f"意图: {r.intent} | 置信度: {r.confidence:.2f} | fallback: {r.fallback_used}")
        if r.reason:
            print(f"理由: {r.reason}")
        if r.fallback_info:
            print(f"补救: {r.fallback_info}")
        print()


if __name__ == "__main__":
    sys.exit(main())
