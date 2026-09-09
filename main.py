"""小旦答命令行交互入口（开发调试用；正式入口见 deployment/api.py）。

用法：python main.py
前置条件：uv sync（或在已激活的 .venv 中），并在 .env 中配置 DEEPSEEK_API_KEY
"""
from agent.graph import invoke


def main() -> None:
    """简单的 REPL 循环：输入问题，输出 Agent 的最终回答。"""
    print("=" * 60)
    print("小旦答：复旦校园智能问答与心理健康监测系统（CLI 模式）")
    print("输入 exit 退出；输入 demo 走内置高危测试样例")
    print("=" * 60)

    session_id = "cli_session"
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "demo":
            # 内置高危样例：验证情绪检测 → 上报 → 关怀回复链路
            user_input = "活着好累啊，感觉撑不下去了"

        result = invoke(
            user_input=user_input,
            user_id="cli_user",
            session_id=session_id,
        )
        print(f"\n小旦答: {result['final_response']}")

        # 开发调试：打印本轮情绪检测结果
        emotion = result.get("emotion")
        if emotion is not None:
            print(f"[debug] 情绪: {emotion.level} (来源: {emotion.source}, "
                  f"置信度: {emotion.confidence:.2f})")


if __name__ == "__main__":
    main()
