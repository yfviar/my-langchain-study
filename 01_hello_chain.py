"""
================================================================================
01 - LangChain 的第一个链条（深入版）
================================================================================

目标不是"跑通代码"，而是理解 LangChain 的设计思想。

读完这段代码你应该能回答:
  1. LangChain 的 | 符号到底做了什么？为什么不用函数调用？
  2. Model 返回的是什么？不是字符串，那是啥？
  3. 为什么要 StrOutputParser？不用它行不行？
  4. PromptTemplate 怎么验证输入？传错变量名会怎样？
  5. invoke / stream / batch 分别适合什么场景？
  6. 如果出错，怎么排查是哪个环节的问题？
  7. 如何换个模型？换个 API 地址？为什么不需要改调用代码？

================================================================================
先解决一个关键误解: LangChain 不是一个"调用库"，而是一个"组合框架"
================================================================================

普通的 OpenAI 调用逻辑:
  client = OpenAI(api_key="...", base_url="...")
  response = client.chat.completions.create(model="gpt-4o", messages=[...])
  text = response.choices[0].message.content

LangChain 做的是另外一件事:
  - 把 "构造提示词" 抽象成 PromptTemplate
  - 把 "调用模型" 抽象成 ChatModel
  - 把 "解析输出" 抽象成 OutputParser
  - 用 | (管道) 把它们串联起来

这带来了两个好处:
  1. 换模型: 只需改配置，Prompt 和 Parser 不用动
  2. 可组装: 可以插入中间处理步骤，像搭积木一样

================================================================================
核心概念: Runnable 接口
================================================================================

LangChain 中所有可以"执行"的东西都实现了 Runnable 接口。

Runnable 提供三种基础执行模式:
  - invoke(input)   -> 同步执行，返回完整结果
  - stream(input)   -> 逐块产出，内存友好
  - batch(inputs)   -> 批量执行，适合离线处理

以及对应的异步版本: ainvoke, astream, abatch

ChatOpenAI、PromptTemplate、StrOutputParser 都是 Runnable。

关键是: 两个 Runnable 用 | 串联后，得到的还是一个 Runnable。
这就是"组合性"(composability) —— 你可以无限串联。
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()


# ============================================================================
# 第一部分: 认识最基础的单元 —— ChatModel
# ============================================================================

def demo_1_raw_model():
    """
    直接调用 ChatOpenAI，看看它到底返回什么。

    关键认知:
      ChatOpenAI.invoke() 返回的是 AIMessage 对象，不是字符串！
      AIMessage 的结构: { content: str, response_metadata: {...}, ... }
      .content 属性才是你想要的文本。

    这也是为什么后面需要 StrOutputParser ——
    它的唯一作用就是提取 .content。
    """
    print("=" * 60)
    print("【第一部分】认识 ChatModel 的输入输出")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        # temperature 控制随机性, 0 = 确定性, 1 = 最大随机
        temperature=0.7,
        # max_completion_tokens 控制最大输出长度
        max_completion_tokens=100,
    )

    # 方式一: 传入字符串 —— LangChain 自动包装为 HumanMessage
    response = model.invoke("用一句话介绍江南的天气")
    print(f"\n[invoke 返回值类型] {type(response).__name__}")
    print(f"[response 是 AIMessage 吗?] {isinstance(response, AIMessage)}")
    print(f"[.content] {response.content}")
    print(f"[.response_metadata] {response.response_metadata}")

    # 方式二: 传入消息列表 —— 精确控制每条消息的角色
    messages = [
        SystemMessage(content="你是一位气象学家，回答要专业。"),
        HumanMessage(content="江南为什么多雨？"),
    ]
    response = model.invoke(messages)
    print(f"\n[SystemMessage 方式] {response.content[:100]}...")


def demo_2_raw_stream():
    """
    流式调用: 为什么要用流式？

    非流式: 发送请求 -> 等待全部生成完 -> 一次性返回 (可能等 10 秒)
    流式:   发送请求 -> 生成一个字就返回一个字 (打字机效果)

    原理:
      ChatOpenAI 构造函数传入 streaming=True
      然后用 for chunk in model.stream(...) 逐块接收

    注意:
      stream() 本身返回的是一个迭代器
      每个 chunk 仍然是 AIMessage，需要 .content 才能拿到文本
      有些 chunk 的 content 可能是空字符串 (比如最后一个)
    """
    print("\n" + "=" * 60)
    print("【第二部分】流式输出")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        streaming=True,
    )

    print("[流式] 用三句话介绍 Python: ", end="", flush=True)
    for chunk in model.stream("用三句话介绍 Python"):
        # chunk 是 AIMessage, chunk.content 是本次的增量文本
        print(chunk.content, end="", flush=True)
    print("\n")


# ============================================================================
# 第二部分: 理解 PromptTemplate —— 为什么不能直接拼字符串
# ============================================================================

def demo_3_prompt_template():
    """
    为什么不用 f-string？

    比如有人这样写:
      f\"你是一个{role}，请用{style}的方式回答: {question}\"

    问题:
      1. 不可复用: 每次都要重新写 f-string
      2. 不安全: 如果 {question} 包含恶意指令怎么办？
         （prompt injection 风险）
      3. 不可追踪: 你不知道最终发给模型的 prompt 长什么样
      4. 不支持消息历史: f-string 只能生成纯文本，不能区分 system/user/assistant

    PromptTemplate 解决了:
      1. 定义一次，反复使用 (invoke 时传入变量)
      2. 自动验证: 传错变量名会报 KeyError，不会静默失败
      3. 支持消息角色: ChatPromptTemplate 可以区分 system/user/assistant
      4. 可组合: 两个 PromptTemplate 可以用 + 合并
    """
    print("=" * 60)
    print("【第三部分】PromptTemplate vs f-string")
    print("=" * 60)

    # 简单模板: 一个变量
    from langchain_core.prompts import PromptTemplate
    template = PromptTemplate.from_template("把'{term}'翻译成英文, 只返回翻译结果")
    prompt_text = template.format(term="人工智能")
    print(f"[模板输出] {prompt_text}")

    # Chat 模板: 区分角色
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("user", "{question}"),
    ])
    messages = chat_prompt.format_messages(
        role="Python 专家",
        question="Python 的 GIL 是什么？"
    )
    print(f"\n[Chat 模板输出]")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content[:80]}...")

    # 错误场景: 缺少变量
    try:
        chat_prompt.format_messages(role="Python 专家")
    except KeyError as e:
        print(f"\n[变量缺失报错] KeyError: {e} —— 防止静默失败")


# ============================================================================
# 第三部分: StrOutputParser —— 它是必要的吗？
# ============================================================================

def demo_4_output_parser():
    """
    回顾: model.invoke() 返回 AIMessage, 不是 str。
    如果你直接在 chain 里用 model 的输出作为下一个环节的输入，
    会得到一个 AIMessage 对象而不是纯文本。

    StrOutputParser 的作用就一件事: 提取 .content 属性。

    LangChain 还提供了其他 Parser:
      - JsonOutputParser: 解析 JSON -> dict
      - PydanticOutputParser: 解析为 Pydantic 对象 (类型安全)
      - CommaSeparatedListOutputParser: 解析逗号分隔的列表
      - XMLOutputParser: 解析 XML

    所以 Parser 的本质是: 把模型返回的非结构化内容，
    转换成下游代码可以直接消费的结构化数据。
    """
    print("=" * 60)
    print("【第四部分】为什么需要 StrOutputParser")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 没有 Parser: 返回 AIMessage
    response = model.invoke("说'你好'")
    print(f"[无 Parser] 类型: {type(response).__name__}")
    print(f"[无 Parser] 内容: {response.content}")
    print(f"[无 Parser] 如果当字符串用: response + '追加文本' 会报错")

    # 有 Parser: 返回纯字符串
    parser = StrOutputParser()
    result = parser.invoke(response)  # 本质: return response.content
    print(f"\n[有 Parser] 类型: {type(result).__name__}")
    print(f"[有 Parser] 内容: {result}")


# ============================================================================
# 第四部分: | 管道的魔力 —— LCEL (LangChain Expression Language)
# ============================================================================

def demo_5_the_pipe():
    """
    这才是 LangChain 最核心的设计。

    chain = prompt | model | parser

    这句代码等同于:
      RunnableSequence(
        first=prompt,          # 第一步: 格式化提示词
        middle=[model],         # 第二步: 调用模型
        last=parser             # 第三步: 解析输出
      )

    数据流:
      {"role": "老师", "question": "什么是多态"}  ← invoke 的输入
        ↓ prompt.invoke(input)
      [SystemMessage("你是一个老师"), HumanMessage("什么是多态")]
        ↓ model.invoke(messages)
      AIMessage(content="多态是指...")
        ↓ parser.invoke(aimessage)
      "多态是指..."  ← 最终输出, 纯字符串

    每步的输入类型和输出类型:
      prompt:   dict -> List[Message]   (模板变量 -> 消息列表)
      model:    List[Message] -> AIMessage  (消息 -> 模型响应)
      parser:   AIMessage -> str         (提取 content)

    为什么用 | 而不是函数调用？
      1. 声明式: 读代码就能看到完整流程
      2. 可插拔: 中间插入/替换步骤很方便
      3. 自动类型检查: 前后步骤类型不匹配会报错
      4. 统一接口: chain 本身也是 Runnable, 可以用 invoke/stream/batch
    """
    print("=" * 60)
    print("【第五部分】| 管道的本质")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，用{style}的方式回答问题。"),
        ("user", "{question}"),
    ])
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    parser = StrOutputParser()

    # 串联
    chain = prompt | model | parser

    # 查看 chain 的结构
    print(f"[chain 类型] {type(chain).__name__}")
    print(f"[chain 输入 schema] {chain.input_schema.model_json_schema()}")
    print(f"[chain 输出 schema] {chain.output_schema.model_json_schema()}")

    # 执行
    result = chain.invoke({
        "role": "小学老师",
        "style": "用打比方的方式",
        "question": "什么是互联网？",
    })
    print(f"\n[执行结果] {result[:100]}...")


# ============================================================================
# 第五部分: invoke vs stream vs batch
# ============================================================================

def demo_6_execution_modes():
    """
    三种执行模式的本质区别:

    invoke:  适合单次调用，等全部结果返回
             优点: 简单
             缺点: 长回复要等很久

    stream:  适合 UI 展示（打字机效果）
             优点: 用户体验好，不用干等
             缺点: 处理流式数据稍微复杂

    batch:   适合批量处理多个输入
             优点: 可以并发执行，效率高
             缺点: 一次请求量大，可能触发限流

    关键点: 同一个 chain, 三种调用方式都支持。
    这说明 chain 和 model 一样，也是一个完整的 Runnable。
    """
    print("\n" + "=" * 60)
    print("【第六部分】invoke / stream / batch")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template("将'{text}'翻译成英文, 只返回结果")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    chain = prompt | model | StrOutputParser()

    # invoke: 单次调用
    result = chain.invoke({"text": "人工智能"})
    print(f"[invoke] {result}")

    # stream: 流式 (对翻译这种短文本, 流式意义不大, 但机制是一样的)
    print("[stream] ", end="", flush=True)
    for chunk in chain.stream({"text": "机器学习"}):
        print(chunk, end="", flush=True)
    print()

    # batch: 批量
    results = chain.batch([
        {"text": "深度学习"},
        {"text": "自然语言处理"},
        {"text": "计算机视觉"},
    ])
    print(f"[batch] {results}")


# ============================================================================
# 第六部分: 调试 —— 出问题了怎么看
# ============================================================================

def demo_7_debug():
    """
    常见问题和排查方法:

    1. 模型返回空字符串
       → 检查 max_tokens 是否太小
       → 检查 prompt 是否太模糊

    2. 变量名写错
       → LangChain 会抛出 KeyError, 明确指出哪个变量缺失

    3. 网络错误
       → 检查 base_url 是否正确
       → 检查 API Key 是否有效

    4. 输出格式不对
       → 用 callback / with_fallbacks() 捕获中间输出

    5. 想查看中间结果
       → 把 chain 拆开来调试:
         prompt_result = prompt.invoke(input)
         model_result = model.invoke(prompt_result)
         final = parser.invoke(model_result)
       → 或者用 chain.with_config(tags=["debug"]) 加回调

    LangSmith 是官方推荐的调试平台, 但也可以本地调试。
    """
    print("=" * 60)
    print("【第七部分】调试技巧")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{role}。"),
        ("user", "{question}"),
    ])
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    parser = StrOutputParser()
    chain = prompt | model | parser

    # 调试技巧 1: 拆开 chain, 逐步查看中间结果
    print("[调试] 逐步查看每个环节的输出:")

    step1 = prompt.invoke({"role": "翻译官", "question": "把'我很快乐'翻译成英文"})
    print(f"  Step1 (prompt 输出): {step1}")

    step2 = model.invoke(step1)
    print(f"  Step2 (model 输出): {step2}")
    print(f"  Step2 (类型检查): {type(step2).__name__}")

    step3 = parser.invoke(step2)
    print(f"  Step3 (parser 输出): {step3}")

    # 调试技巧 2: 多次调用同一 chain 结果可能略有不同
    direct_result = chain.invoke({
        "role": "翻译官",
        "question": "把'我很快乐'翻译成英文",
    })
    print(f"\n  [对比] 直接调用结果: {direct_result}")
    print(f"  [说明] LLM 每次调用的输出可能不同，这是正常的")


# ============================================================================
# 第七部分: 异步 —— 生产环境的正确打开方式
# ============================================================================

async def demo_8_async():
    """
    在生产环境 (FastAPI, Flask-Async 等) 中, 应该用异步版本:
      ainvoke, astream, abatch

    为什么？
      HTTP 请求本质是 IO 密集型操作, 用同步模式会阻塞事件循环。
      异步模式下, 等待网络响应时可以处理其他请求。

    要点:
      - 把 invoke 换成 ainvoke
      - 函数声明 async def
      - 调用时用 await
    """
    print("\n" + "=" * 60)
    print("【第八部分】异步调用")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template("一句话介绍{topic}")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    chain = prompt | model | StrOutputParser()

    result = await chain.ainvoke({"topic": "Python asyncio"})
    print(f"[ainvoke] {result[:80]}...")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_raw_model()
    demo_2_raw_stream()
    demo_3_prompt_template()
    demo_4_output_parser()
    demo_5_the_pipe()
    demo_6_execution_modes()
    demo_7_debug()

    import asyncio
    asyncio.run(demo_8_async())

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. ChatOpenAI.invoke() 返回的是 AIMessage 对象，不是字符串
2. StrOutputParser 的作用就是提取 .content，它是必要的
3. | 运算符创建 RunnableSequence，每步输出喂给下一步输入
4. PromptTemplate 不只是拼字符串，它提供变量验证和角色区分
5. invoke 适合单次/开发，stream 适合 UI，batch 适合批处理
6. 调试时把 chain 拆开逐步执行，定位问题环节
7. 同一个 chain 换 model 不需要改代码，只需改配置
8. 生产环境用 ainvoke/astream 避免阻塞事件循环

下一节 02_prompt_template.py 会深入 PromptTemplate 的各种变体。
""")
