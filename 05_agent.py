"""
================================================================================
05 - Agent 智能体深度解析
================================================================================

你会学到:
  1. Agent 和 Chain 的本质区别是什么?
  2. ReAct 模式: 模型如何 "观察 → 思考 → 行动"?
  3. @tool 装饰器: 工具定义的内在要求
  4. Agent 如何决定调用哪个工具? (tool calling 机制)
  5. Agent 的执行循环: 模型思考和行动的边界在哪?
  6. 工具设计原则: 什么应该做成工具, 什么不应该?
  7. Agent 的局限和陷阱

================================================================================
Agent vs Chain: 本质区别

Chain (链条):
  固定的执行流程: A -> B -> C
  设计时就知道每一步做什么
  适合: 翻译、总结、分类等确定性任务

Agent (智能体):
  动态的执行流程: 模型自己决定下一步做什么
  运行时才知道要不要调工具、调哪个工具
  适合: 需要决策和多步骤的复杂任务

Chain 像一条预设的流水线, Agent 像一个会自己决策的工人。
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ============================================================================
# 第一部分: 工具定义 —— @tool 装饰器做了什么
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。输入如: '2+3*4', '(10+5)*3', 'sqrt(16)'

    关键:
      - 这个 docstring 会被发送给模型, 作为工具的 "使用说明"
      - 模型根据 docstring 判断什么时候应该调用这个工具
      - expression 的类型标注 (str) 也会被用来生成工具的输入 schema
    """
    try:
        import math
        safe_dict = {
            "__builtins__": None,
            "math": math, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        }
        result = eval(expression, safe_dict)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_knowledge(query: str) -> str:
    """在公司知识库中搜索信息。参数 query 是搜索关键词或问题。"""
    knowledge = {
        "公司地址": "ACME 科技, 杭州市余杭区未来科技城 ACME 大厦",
        "工作时间": "9:00-18:00, 弹性打卡, 午休 12:00-13:30",
        "请假流程": "在 OA 系统中提交请假申请, 直属上级审批",
        "团建": "每个季度一次团建活动, 由行政部门组织",
        "报销": "在财务系统中提交报销申请, 需要发票和审批",
    }

    for key, value in knowledge.items():
        if key in query or any(w in query for w in key):
            return f"[{key}] {value}"

    return f"未找到关于 '{query}' 的信息"


@tool
def get_datetime() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def demo_1_tool_anatomy():
    """
    @tool 装饰器的本质:

    把普通 Python 函数包装成一个 LangChain Tool 对象。
    Tool 对象包含:
      - name:        工具名称 (函数名)
      - description: 工具描述 (docstring)
      - args_schema: 输入 schema (从类型标注自动生成)
      - func:        实际执行的函数

    Agent 通过 Tool 的 description 和 args_schema 来理解:
      - 这个工具是干什么的
      - 什么时候应该用它
      - 需要传什么参数

    所以, docstring 写得越清楚, Agent 就越不容易用错工具。
    """
    print("=" * 60)
    print("【第一部分】工具定义的解剖")
    print("=" * 60)

    # 查看 @tool 到底生成了什么
    print(f"[calculator 类型] {type(calculator).__name__}")
    print(f"[calculator.name] {calculator.name}")
    print(f"[calculator.description] {calculator.description[:80]}...")
    print(f"[calculator.args_schema] {calculator.args_schema.model_json_schema()}")
    print()

    # 直接调用 (不走 Agent)
    result = calculator.invoke({"expression": "2+3*4"})
    print(f"[直接调用 calculator] {result}")


# ============================================================================
# 第二部分: ReAct 模式 —— 思考与行动交替
# ============================================================================

def demo_2_react():
    """
    ReAct = Reasoning + Acting

    Agent 的执行循环:

    1. 观察 (Observe): Agent 收到用户消息
    2. 思考 (Reason): 模型分析 "我该做什么? 要不要用工具? 用哪个?"
    3. 行动 (Act):   调用工具, 得到结果
    4. 观察 (Observe): 把工具结果发回模型
    5. 回到步骤 2 循环, 直到模型决定 "可以直接回答了"
    6. 回答: 基于所有信息生成最终回复

    这个模式的关键:
      模型在每一步都可以重新决策。
      它不是一次性决定所有工具调用, 而是边做边看。

    从消息层面看, Agent 内部的消息流是:
      [System(prompt)] -> [User(问题)] -> [AI(要调用工具)] ->
      [Tool(工具结果)] -> [AI(要调用另一个工具)] -> [Tool(结果)] ->
      [AI(最终回答)]
    """
    print("\n" + "=" * 60)
    print("【第二部分】ReAct 模式的执行循环")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    agent = create_agent(
        model,
        [calculator, search_knowledge, get_datetime],
    )

    # 一个需要两步工具调用的任务
    task = "现在是什么时间? 然后帮我算一下, 今年的年份除以 4 等于多少?"

    print(f"[用户] {task}")
    result = agent.invoke({"messages": [HumanMessage(content=task)]})

    # 遍历中间步骤, 展示 Agent 的思考过程
    print("\n[Agent 的执行过程]")
    for i, msg in enumerate(result["messages"]):
        role = type(msg).__name__
        content = msg.content[:100] if hasattr(msg, 'content') and msg.content else ""
        print(f"  [{i}] {role}: {content}")

    print(f"\n[最终回答] {result['messages'][-1].content[:200]}...")


# ============================================================================
# 第三部分: Agent 如何选择工具?
# ============================================================================

def demo_3_tool_selection():
    """
    Agent 怎么决定调用哪个工具?

    1. 系统 prompt 里列出了所有工具的描述
    2. 模型看用户的问题
    3. 模型判断: 哪个工具的 description 和用户问题最匹配?
    4. 模型生成 tool call 请求 (包含工具名和参数)
    5. Agent 框架解析 tool call, 调用实际函数
    6. 把函数返回值作为 ToolMessage 发回模型

    这整个过程依赖的是模型的 Function Calling 能力。
    不是 LangChain 专有的, 而是 OpenAI/Anthropic 等模型的原生能力。

    所以:
      - 工具描述写得好不好, 直接决定 Agent 能不能正确选择
      - 小模型可能不支持 Function Calling, Agent 就不可用
    """
    print("\n" + "=" * 60)
    print("【第三部分】Agent 如何选择工具")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    agent = create_agent(model, [calculator, search_knowledge, get_datetime])

    # 测试不同类型的请求, 看 Agent 选哪个工具
    tests = [
        ("计算类", "计算 123 * 456"),
        ("知识类", "公司团建怎么安排?"),
        ("时间类", "现在是几点了?"),
        ("模糊类", "帮我看看报销"),
    ]

    for category, task in tests:
        result = agent.invoke({"messages": [HumanMessage(content=task)]})

        # 分析用了哪些工具
        tool_calls = [
            msg for msg in result["messages"]
            if hasattr(msg, "tool_calls") and msg.tool_calls
        ]
        if tool_calls:
            tools_used = [tc["name"] for tc in tool_calls[0].tool_calls]
            print(f"\n[{category}] {task}")
            print(f"  使用的工具: {tools_used}")
        else:
            print(f"\n[{category}] {task} → 直接回答, 未使用工具")


# ============================================================================
# 第四部分: 工具设计原则
# ============================================================================

def demo_4_tool_design():
    """
    好的工具设计:

    1. 单一职责: 一个工具只做一件事
       坏: "查天气和发邮件" (两个功能)
       好: "查天气" 和 "发邮件" 各一个工具

    2. 描述要精确: docstring 要写清输入输出
       坏: "处理数据"
       好: "计算数学表达式, 输入如 '2+3*4', 返回计算结果"

    3. 参数要明确: 类型标注要准确
       坏: def search(x)
       好: def search(query: str)

    4. 返回值要精炼: 不要返回巨量数据给模型
       坏: 返回整个数据库查询结果 (10 万行)
       好: 返回 top 5 条摘要

    5. 有确定性: 相同输入多次返回相同结果
       坏: 每次返回随机结果 (模型会困惑)
       好: 幂等操作

    什么时候不该用工具?
      - 任务是确定性的 (翻译、总结) → Chain 更好
      - 工具太多 (超过 10 个) → 模型选择困难, 容易选错
      - 工具之间功能重叠 → 模型不知道该用哪个
    """
    print("\n" + "=" * 60)
    print("【第四部分】工具设计原则")
    print("=" * 60)
    print("""
设计 Checklist:
  □ 一个工具只做一件事
  □ docstring 写了什么情况下该用这个工具
  □ 参数有明确的类型标注
  □ 返回值不会太大 (控制在 1K 字符以内)
  □ 相同输入多次调用结果一致
  □ 工具名能一眼看出功能
""")

    # 反面教材: 一个描述模糊的工具
    @tool
    def do_stuff(data: str) -> str:
        """处理数据"""
        return data[::-1]  # 反转字符串... 但描述完全没说

    # 这会导致 Agent 不知道该不该用它, 可能误用


# ============================================================================
# 第五部分: Agent 的陷阱
# ============================================================================

def demo_5_pitfalls():
    """
    Agent 常见问题:

    1. 无限循环
       模型调工具 → 不满意 → 再调 → 再调...
       解决: 设置 max_iterations

    2. 错误使用工具
       模型把英文翻译请求发给了计算器工具
       解决: 改进工具描述, 让边界更清楚

    3. 幻觉工具调用
       模型编造了一个不存在的工具名
       解决: 框架会报错, Agent 收到错误后自我修正

    4. 工具结果幻觉
       模型忽略了工具返回的结果, 自己编了一个
       解决: prompt 里强调 "必须基于工具返回的结果回答"

    5. Token 爆炸
       多轮思考 + 多工具调用 = 大量 token 消耗
       解决: 控制最大步数, 精简工具输出
    """
    print("\n" + "=" * 60)
    print("【第五部分】Agent 的常见陷阱")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,  # Agent 任务建议用低 temperature
    )

    agent = create_agent(model, [calculator, search_knowledge])

    # 边界测试: 问一个非工具能解决的问题
    result = agent.invoke({
        "messages": [HumanMessage(content="你觉得 Python 和 JavaScript 哪个好?")]
    })
    print(f"[非工具问题] {result['messages'][-1].content[:200]}...")

    # 边界测试: 问一个两个工具都相关的模糊问题
    result = agent.invoke({
        "messages": [HumanMessage(content="帮我查一下工作时间和今天的日期")]
    })
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_names = [tc["name"] for tc in msg.tool_calls]
            print(f"[多工具调用] 同时调用了: {tc_names}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_tool_anatomy()
    demo_2_react()
    demo_3_tool_selection()
    demo_4_tool_design()
    demo_5_pitfalls()

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. Agent 动态决策, Chain 固定流程, 复杂任务用 Agent
2. ReAct: 观察 → 思考 → 行动 → 观察 → ... → 回答
3. @tool 的 docstring 是模型选择工具的关键依据
4. Agent 靠 Function Calling 机制选择工具
5. 工具设计: 单一职责、描述精确、参数明确、返回值精简
6. Agent 陷阱: 无限循环、误用工具、幻觉调用、Token 爆炸
7. 简单确定任务用 Chain, 动态决策任务用 Agent

下一节 06_memory.py 会深入如何管理多轮对话的上下文记忆。
""")
