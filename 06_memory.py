"""
================================================================================
06 - 对话记忆 (Memory) 深度解析
================================================================================

你会学到:
  1. LLM 为什么需要 "记忆"? (无状态 vs 有状态)
  2. 对话历史的存储和注入机制
  3. InMemoryChatMessageHistory 和 store 模式
  4. RunnableWithMessageHistory: 它如何包装任何 chain?
  5. Session 隔离: 为什么每个用户需要独立的 session?
  6. Token 管理: 对话太长怎么办?
  7. 摘要记忆: 把旧对话压缩成摘要
  8. 生产环境的记忆方案 (Redis, Postgres, 滑动窗口)

================================================================================
核心问题: LLM 是无状态的

每次调用 model.invoke(messages) 都是独立的。
你问 "我叫什么?" -> 模型不知道, 因为上次对话已经结束了。

为了让模型 "记住" 之前说过的话, 你需要:
  1. 把之前的对话保存起来
  2. 每次调用时, 把历史消息和新消息一起发给模型

LangChain 的 Memory 模块就是做这件事的框架。
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


# ============================================================================
# 第一部分: 无记忆 vs 有记忆 —— 直观对比
# ============================================================================

def demo_1_no_memory():
    """
    无记忆模式:
      每次调用都是独立的, 模型不记得上一次说了什么。

    这是 LLM 的默认行为, 也是它的本质限制。
    模型只看到你当前发的消息, 看不到之前的对话。
    """
    print("=" * 60)
    print("【第一部分】无记忆 vs 有记忆")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。"),
        ("user", "{input}"),
    ])
    chain = prompt | model | StrOutputParser()

    conversations = [
        "我叫小明, 今年 25 岁。",
        "我叫什么名字?",
        "我多大了?",
    ]

    print("--- 无记忆 ---")
    for msg in conversations:
        result = chain.invoke({"input": msg})
        print(f"  用户: {msg}")
        print(f"  助手: {result[:80]}...")
        print()

    print("[结论] 模型完全不记得之前的对话")


# ============================================================================
# 第二部分: ChatMessageHistory —— 存储对话的容器
# ============================================================================

def demo_2_chat_history():
    """
    ChatMessageHistory 就是一个消息列表的容器。
    你可以往里面添加消息, 查看历史, 清空历史。

    它是 Memory 模块的底层基础设施。
    所有其他 Memory 方案 (Redis, Postgres, 文件) 都是它的不同实现。

    InMemoryChatMessageHistory:
      - 数据存在内存中
      - 进程重启就没了
      - 适合: 开发、测试、单会话临时记忆

    生产环境替代方案:
      - RedisChatMessageHistory:   存在 Redis, 适合分布式
      - PostgresChatMessageHistory: 存在 PG, 适合持久化
      - FileChatMessageHistory:     存在本地文件, 适合简单场景
    """
    print("\n" + "=" * 60)
    print("【第二部分】ChatMessageHistory 容器")
    print("=" * 60)

    # 创建一个内存历史
    history = InMemoryChatMessageHistory()

    # 模拟对话
    history.add_message(HumanMessage(content="你好, 我叫小明"))
    history.add_message(AIMessage(content="你好小明! 有什么可以帮你的?"))
    history.add_message(HumanMessage(content="帮我查天气"))

    # 获取历史
    messages = history.messages
    print(f"历史消息数: {len(messages)}")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    # 清空历史
    history.clear()
    print(f"\n清空后消息数: {len(history.messages)}")


# ============================================================================
# 第三部分: store 模式 —— 管理多个用户的历史
# ============================================================================

def demo_3_store_pattern():
    """
    现实场景: 系统多个用户同时使用。
    你需要为每个用户维护独立的对话历史。

    store 模式:
      store = {}  # key = session_id, value = ChatMessageHistory

    每个用户/会话有一个 session_id, 对应一个独立的历史记录。

    为什么不能全局共享?
      如果所有用户共用一份历史, 用户 A 说的话会被用户 B 看到。
      这既是隐私问题, 也是功能问题。
    """
    print("\n" + "=" * 60)
    print("【第三部分】Store 模式 —— 多用户隔离")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是订单助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])
    chain = prompt | model | StrOutputParser()

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 用户 A 下单
    print("--- 用户 A (session-a) ---")
    r1 = chain_with_history.invoke(
        {"input": "我的订单号是 A001, 查一下状态"},
        config={"configurable": {"session_id": "session-a"}},
    )
    print(f"  A 说: 我的订单号是 A001, 查一下状态")
    print(f"  助手: {r1[:80]}...")

    r2 = chain_with_history.invoke(
        {"input": "我的订单号是什么?"},
        config={"configurable": {"session_id": "session-a"}},
    )
    print(f"\n  A 说: 我的订单号是什么?")
    print(f"  助手: {r2[:80]}...")

    # 用户 B: 不会有 A 的历史
    print("\n--- 用户 B (session-b) ---")
    r3 = chain_with_history.invoke(
        {"input": "我的订单号是什么?"},
        config={"configurable": {"session_id": "session-b"}},
    )
    print(f"  B 说: 我的订单号是什么?")
    print(f"  助手: {r3[:80]}...")


# ============================================================================
# 第四部分: RunnableWithMessageHistory 的工作原理
# ============================================================================

def demo_4_runnable_with_history():
    """
    RunnableWithMessageHistory 的内部机制:

    它包装了任何实现了 Runnable 接口的 chain, 并执行以下逻辑:

    1. 接收 invoke(input, config)
    2. 从 config 中提取 session_id
    3. 调用 get_session_history(session_id) 获取历史
    4. 从历史中读取所有消息, 注入到 prompt 的 "history" 位置
    5. 调用原始的 chain
    6. 把本轮对话 (HumanMessage + AIMessage) 追加到历史中
    7. 返回结果

    关键参数:
      input_messages_key:  哪个变量接收用户输入 (默认 "input")
      history_messages_key: 哪个变量接收历史消息 (默认 "history")
      output_messages_key: 哪个变量接收模型输出 (默认 "output")

    这意味着它可以包装任何 chain, 不关心 chain 内部是怎么实现的。
    只要 chain 的 prompt 有 MessagesPlaceholder("history"), 就能注入历史。
    """
    print("\n" + "=" * 60)
    print("【第四部分】RunnableWithMessageHistory 原理")
    print("=" * 60)

    print("""
RunnableWithMessageHistory 的调用流程:

  chain.invoke({"input": "你好"}, config={"session_id": "abc"})
    ↓
  1. get_session_history("abc") → [HumanMessage("我叫小明"), AIMessage("你好小明")]
    ↓
  2. 构造新输入: {"input": "你好", "history": [HumanMessage, AIMessage]}
    ↓
  3. 调用原始 chain → "你好! 有什么可以帮助小明的?"
    ↓
  4. 追加到历史: + HumanMessage("你好") + AIMessage("你好! ...")
    ↓
  5. 返回 "你好! 有什么可以帮助小明的?"
""")


# ============================================================================
# 第五部分: Token 管理 —— 对话太长怎么办?
# ============================================================================

def demo_5_token_management():
    """
    对话越来越长, 最终会超出模型的上下文窗口。

    解决策略:

    1. 滑动窗口 (Trim)
       只保留最近 N 条消息, 丢弃旧的
       优点: 简单, 实现容易
       缺点: 丢失早期重要信息

    2. 摘要 (Summarize)
       把旧的对话用模型压缩成一段摘要
       优点: 保留关键信息, 节省 token
       缺点: 摘要可能遗漏细节

    3. 混合 (摘要 + 窗口)
       旧对话 → 摘要
       最近 N 条 → 保留原文
       两者拼接

    4. 向量检索 (Vector Memory)
       把所有历史消息向量化, 需要时检索最相关的
       优点: 不丢任何信息
       缺点: 复杂度高
    """
    print("\n" + "=" * 60)
    print("【第五部分】Token 管理策略")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 模拟一段长对话
    full_history = [
        HumanMessage(content="我叫小明, 住在北京"),
        AIMessage(content="你好小明!"),
        HumanMessage(content="我想学 Python"),
        AIMessage(content="很好的选择! Python 适合入门。"),
        HumanMessage(content="我之前学过一点 JavaScript"),
        AIMessage(content="那你会学得很快, Python 和 JS 有不少相似之处。"),
        HumanMessage(content="有什么好的学习项目推荐?"),
        AIMessage(content="推荐做一个 Todo List 应用, 很经典。"),
        HumanMessage(content="我现在住在上海了"),
        AIMessage(content="哦, 你搬去上海了?"),
    ]

    print(f"完整历史: {len(full_history)} 条消息")

    # 策略 1: 滑动窗口 (只保留最后 4 条)
    window_history = full_history[-4:]
    print(f"滑动窗口: {len(window_history)} 条")

    # 策略 2: 摘要
    summary_prompt = ChatPromptTemplate.from_template(
        "将以下对话总结为一段简短摘要:\n\n{conversation}"
    )
    summarizer = summary_prompt | model | StrOutputParser()

    conversation_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in full_history
    )
    summary = summarizer.invoke({"conversation": conversation_text})

    print(f"\n策略 2 - 摘要:")
    print(f"  原始: {len(conversation_text)} 字符")
    print(f"  摘要: {len(summary)} 字符")
    print(f"  内容: {summary[:150]}...")

    # 策略 3: 混合 (摘要 + 最近 4 条)
    truncated = full_history[:-4]  # 旧的用摘要
    recent = full_history[-4:]     # 最新的保留原文

    truncated_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in truncated
    )
    truncated_summary = summarizer.invoke({"conversation": truncated_text})

    print(f"\n策略 3 - 混合:")
    print(f"  摘要部分: {truncated_summary}")
    print(f"  保留原文: {len(recent)} 条最新消息")


# ============================================================================
# 第六部分: 摘要记忆实战
# ============================================================================

def demo_6_summary_memory():
    """
    摘要记忆的完整流程:

    1. 维护对话历史和摘要
    2. 每次对话后:
       - 如果历史太长, 触发摘要
       - 把旧消息压缩成摘要
       - 新消息追加到历史
    3. 构建 prompt 时:
       - 系统消息 + 摘要 + 最近的消息

    LangChain 的 ConversationSummaryBufferMemory 做了这件事,
    但在 LangGraph 时代更推荐手动控制。
    """
    print("\n" + "=" * 60)
    print("【第六部分】摘要记忆实战")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    summary_prompt = ChatPromptTemplate.from_template(
        "将以下对话逐条总结为简短摘要, 保留关键信息(姓名、地点、事实):\n\n{conversation}"
    )
    summarizer = summary_prompt | model | StrOutputParser()

    # 构建带摘要的记忆链
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手。以下是之前对话的摘要:\n\n{summary}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])
    memory_chain = memory_prompt | model | StrOutputParser()

    store = {}
    summary_store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    def get_summary(session_id: str) -> str:
        return summary_store.get(session_id, "")

    def update_summary(session_id: str, history: BaseChatMessageHistory):
        if len(history.messages) >= 6:
            conversation = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in history.messages
            )
            summary_store[session_id] = summarizer.invoke({"conversation": conversation})

    chain_with_memory = RunnableWithMessageHistory(
        memory_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 模拟多轮对话
    conversations = [
        "我叫张三, 今年 30 岁, 住在杭州。",
        "我的工作是软件工程师。",
        "我养了一只猫叫咪咪。",
        "我之前问过你什么?",
    ]

    for i, msg in enumerate(conversations):
        summary = get_summary("demo")
        result = chain_with_memory.invoke(
            {"input": msg, "summary": summary},
            config={"configurable": {"session_id": "demo"}},
        )
        update_summary("demo", store["demo"])
        print(f"[轮次 {i+1}] 用户: {msg}")
        print(f"         摘要: {summary[:60] or '(空)'}...")
        print(f"         助手: {result[:80]}...")
        print()


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_no_memory()
    demo_2_chat_history()
    demo_3_store_pattern()
    demo_4_runnable_with_history()
    demo_5_token_management()
    demo_6_summary_memory()

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. LLM 无状态, 每次调用独立; Memory 保存历史并注入到 prompt
2. ChatMessageHistory 是消息列表容器, InMemory 适合开发
3. Store 模式用 session_id 隔离不同用户的历史
4. RunnableWithMessageHistory 包装任何 chain, 自动注入/保存历史
5. 滑动窗口、摘要、混合模式是解决 Token 限制的三种策略
6. 摘要记忆把旧对话压缩, 节约 token 的同时保留关键信息
7. 生产环境用 Redis/Postgres 代替 InMemory 存储

============================================================

恭喜! 你已学完 LangChain 六大核心模块:
  01 - 链条基础 (Runnable, |, invoke/stream/batch)
  02 - 提示模板 (角色、历史、Few-shot、Partial)
  03 - 输出解析 (Str/Json/Pydantic, 容错, 自定义)
  04 - RAG (加载→分割→向量化→检索→生成)
  05 - Agent (工具定义、ReAct、Function Calling)
  06 - 记忆 (历史管理、Session 隔离、Token 优化)

接下来可以深入:
  - LangGraph: 自定义状态图和复杂工作流
  - 生产级 RAG: 混合检索、Rerank、查询改写
  - MCP: 给 Agent 接入标准化的外部工具
""")
