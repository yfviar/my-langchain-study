"""
================================================================================
02 - 提示模板 (PromptTemplate) 深度解析
================================================================================

你会学到:
  1. PromptTemplate 和 ChatPromptTemplate 的区别和使用场景
  2. 三种消息角色的含义: system / user / assistant
  3. MessagesPlaceholder: 如何动态插入历史消息
  4. FewShotPromptTemplate: 教模型按格式回答
  5. Partial Prompt: 提前绑定部分变量
  6. Chain of Thought 提示技巧
  7. 在实际项目中怎么组织提示词

================================================================================
核心概念: 消息角色 (Message Roles)
================================================================================

LangChain 把每条消息都建模成一个 Message 对象, 区分角色:

  SystemMessage:    系统提示, 设定 AI 的行为、角色、规则
                    放在对话最前面, 模型会"遵守"这些指令

  HumanMessage:     用户说的话, 来自真实用户
                    每次对话中, 告诉模型用户问了什么

  AIMessage:        模型之前的回复
                    用来维护对话历史, 让模型知道之前说过什么

  ToolMessage:      工具调用的结果 (在 Agent 一节会深入)
                    模型调用工具后, 工具返回的结果

为什么需要区分角色?
  模型在训练时就区分了这些角色, 知道 SystemMessage 是指令、
  HumanMessage 是用户输入、AIMessage 是历史上下文。
  如果混用, 模型的行为会变得不可控。

================================================================================
PromptTemplate vs ChatPromptTemplate
================================================================================

PromptTemplate:      生成纯文本字符串
                     适合: 老式 Completion API, 或者作为中间步骤

ChatPromptTemplate:  生成 Message 列表
                     适合: Chat Completion API (GPT-4, Claude 等)
                     推荐: 这是现代 LLM 的主流用法
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


# ============================================================================
# 第一部分: PromptTemplate vs ChatPromptTemplate
# ============================================================================

def demo_1_two_templates():
    """
    两种模板的核心区别:

    PromptTemplate.from_template("...{var}...")
      -> 返回一个纯文本字符串
      -> 用法: template.format(var="value")
      -> 适合: 作为中间步骤, 拼进更大的模板

    ChatPromptTemplate.from_messages([...])
      -> 返回一个 Message 列表
      -> 用法: template.format_messages(var="value")
      -> 适合: 直接喂给 ChatModel

    什么时候用哪个?
      - 如果你用的是 ChatOpenAI / ChatAnthropic (Chat 模型), 用 ChatPromptTemplate
      - 如果你在构造嵌套模板, 内层可以用 PromptTemplate
    """
    print("=" * 60)
    print("【第一部分】PromptTemplate vs ChatPromptTemplate")
    print("=" * 60)

    # PromptTemplate: 生成纯文本
    simple = PromptTemplate.from_template("将'{word}'翻译成英文")
    result = simple.format(word="人工智能")
    print(f"[PromptTemplate] 类型: {type(result).__name__}")
    print(f"[PromptTemplate] 内容: {result}")

    # ChatPromptTemplate: 生成消息列表
    chat = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手"),
        ("user", "将'{word}'翻译成英文"),
    ])
    messages = chat.format_messages(word="人工智能")
    print(f"\n[ChatPromptTemplate] 类型: {type(messages).__name__}")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")


# ============================================================================
# 第二部分: 消息角色详解
# ============================================================================

def demo_2_message_roles():
    """
    三种角色在模型眼中的含义:

    system:  "你应该遵守这些规则..."
             设定 AI 的身份、风格、行为边界、输出格式
             通常放在 messages 列表的第一条

    user:    "用户问: ..."
             来自用户的输入, 是模型要回答的对象
             一个对话通常由多条 user/assistant 交替组成

    assistant: "AI 之前的回答: ..."
             模型自己之前说过的话
             在维护对话历史时, 需要把 AI 的回复也传回去

    关键认知:
      带历史对话时, 消息列表可能是:
      [system, user1, assistant1, user2, assistant2, user3]
      模型会根据这段历史来理解上下文, 生成 assistant3
    """
    print("\n" + "=" * 60)
    print("【第二部分】消息角色详解")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 场景: 一个完整的对话模拟
    messages = [
        SystemMessage(content="你是一个数学老师, 每次回答都先写'【解题思路】', 再写答案。"),
        HumanMessage(content="3 的平方是多少?"),
        AIMessage(content="【解题思路】3 的平方就是 3 × 3, 也就是 3 乘自己一次。\n答案是 9。"),
        HumanMessage(content="那 3 的立方呢?"),
    ]

    response = model.invoke(messages)
    print("[带历史的对话]")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {type(msg).__name__}: {msg.content[:60]}...")
    print(f"  [回答] {response.content[:100]}...")


# ============================================================================
# 第三部分: MessagesPlaceholder —— 动态插入历史
# ============================================================================

def demo_3_placeholder():
    """
    MessagesPlaceholder 是什么?

    模板中的 {history} 不是一个字符串, 而是一个消息列表。
    它允许你在调用时动态插入之前的对话历史。

    使用场景:
      - 多轮对话: 每次调用都传入之前的所有消息
      - 条件对话: 有时需要历史, 有时不需要
      - 插入检索结果: 把 RAG 的检索结果作为消息插入

    与普通变量的区别:
      ChatPromptTemplate.from_messages([
          ("system", "你是一个助手"),
          ("user", "{input}"),       # ← 这是普通变量, 展开为字符串
          MessagesPlaceholder("history"),  # ← 这是占位符, 插入消息列表
      ])

    注意: MessagesPlaceholder 的参数名和调用时的变量名必须一致。
    """
    print("\n" + "=" * 60)
    print("【第三部分】MessagesPlaceholder")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是购物助手, 帮用户查订单。"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    # 调用 1: 用户告诉助手订单号
    history = []
    result1 = model.invoke(prompt.format_messages(
        history=history,
        input="我的订单号是 A123456, 帮我查一下状态",
    ))
    print(f"[第 1 轮] AI: {result1.content[:80]}...")

    # 调用 2: 引用之前的订单号 (把第 1 轮的问答加入历史)
    history = [
        HumanMessage(content="我的订单号是 A123456, 帮我查一下状态"),
        AIMessage(content=result1.content),
    ]
    result2 = model.invoke(prompt.format_messages(
        history=history,
        input="我刚才说的订单号是什么?",
    ))
    print(f"[第 2 轮] AI: {result2.content[:80]}...")

    # 调用 3: 不传历史 (空列表), 模型不知道之前说了什么
    result3 = model.invoke(prompt.format_messages(
        history=[],
        input="我刚才说的订单号是什么?",
    ))
    print(f"[第 3 轮(无历史)] AI: {result3.content[:80]}...")


# ============================================================================
# 第四部分: FewShotPromptTemplate —— 用示例教会模型
# ============================================================================

def demo_4_few_shot():
    """
    Few-shot 是什么?

    给模型看几个"输入 -> 正确输出"的示例, 然后让它对新输入
    按同样的格式生成输出。这是最古老的 prompt 技巧之一,
    但至今仍然有效。

    原理:
      FewShotPromptTemplate 会在最终的 prompt 里这样构造:
        以下是几个示例:
        示例 1: 输入 -> 输出
        示例 2: 输入 -> 输出
        请按以上格式, 回答: {新输入}

    FewShotPromptTemplate 的三个参数:
      examples:      示例列表, 每个示例是一个 dict
      example_prompt: 如何格式化每个示例 (也是 PromptTemplate)
      prefix:        示例之前的说明文字
      suffix:        示例之后的提问 (包含 {input} 变量)

    适用场景:
      - 需要特定输出格式 (JSON schema, Markdown, 表格...)
      - 做分类任务 (给几个正例负例)
      - 翻译、总结等需要一致风格的场景
    """
    print("\n" + "=" * 60)
    print("【第四部分】FewShotPromptTemplate")
    print("=" * 60)

    # 示例: 把非结构化的电影描述变成结构化 JSON
    # 注意: output 中的 { 和 } 要用 {{ 和 }} 转义, 避免被 format() 当变量
    examples = [
        {
            "input": "流浪地球, 科幻片, 吴京演的, 豆瓣 7.9, 2019 年上映",
            "output": '{{"title": "流浪地球", "genre": "科幻", "star": "吴京", "rating": 7.9, "year": 2019}}',
        },
        {
            "input": "你好李焕英, 喜剧片, 贾玲张小斐, 评分 8.1, 2021 年的",
            "output": '{{"title": "你好李焕英", "genre": "喜剧", "star": "贾玲、张小斐", "rating": 8.1, "year": 2021}}',
        },
    ]

    example_prompt = PromptTemplate.from_template(
        "输入: {input}\n输出: {output}"
    )

    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="将以下电影描述转换为 JSON 格式。参考示例:",
        suffix="输入: {input}\n输出:",
        input_variables=["input"],
    )

    # 看生成的完整 prompt
    full_prompt = few_shot.format(input="长津湖, 战争片, 吴京易烊千玺, 7.4 分, 2021")
    print("[生成的 Few-shot Prompt]")
    print(full_prompt)

    # 让模型执行
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = model.invoke(full_prompt)
    print(f"\n[模型输出] {response.content}")


# ============================================================================
# 第五部分: Partial Prompt —— 提前绑定部分变量
# ============================================================================

def demo_5_partial():
    """
    Partial 是什么?

    有时候你会定义一个有多个变量的模板, 但某些变量在创建时就知道值了。
    这时可以用 .partial() 提前绑定, 调用时只需要传剩下的变量。

    场景:
      system_prompt = "你是{role}, 回答用{language}"
      在中文应用中, language 总是 "中文"
      在英文应用中, language 总是 "英文"

    用 partial 的好处:
      1. 减少重复: 不用每次调用都传 language
      2. 防止遗漏: 绑定的变量不会忘记传
      3. 分层抽象: 上一层绑定通用变量, 下一层传具体内容

    什么时候不要用 partial:
      - 变量确实每次都不同 (如 user input)
      - 需要运行时动态决定值
    """
    print("\n" + "=" * 60)
    print("【第五部分】Partial Prompt")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    base_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{role}, 用{language}回复, 风格: {style}"),
        ("user", "{question}"),
    ])

    # 方式 1: 直接 partial
    zh_prompt = base_prompt.partial(language="中文")
    en_prompt = base_prompt.partial(language="英文")

    zh_result = model.invoke(zh_prompt.format_messages(
        role="老师", style="通俗易懂", question="什么是 API?"
    ))
    en_result = model.invoke(en_prompt.format_messages(
        role="teacher", style="simple and clear", question="What is an API?"
    ))

    print(f"[Partial (中文)] {zh_result.content[:80]}...")
    print(f"[Partial (英文)] {en_result.content[:80]}...")

    # 方式 2: 在整个 chain 上 partial (提前绑定某些变量到 chain 上)
    chain = base_prompt.partial(role="老师", language="中文") | model | StrOutputParser()
    result = chain.invoke({"style": "简单", "question": "什么是递归?"})
    print(f"\n[Chain 上的 Partial] {result[:80]}...")


# ============================================================================
# 第六部分: 深入 from_messages 的三种写法
# ============================================================================

def demo_6_message_formats():
    """
    ChatPromptTemplate.from_messages() 支持三种写法:

    1. 元组写法 (最常用):
       ("system", "你是{role}")
       ("user", "{question}")
       第一个元素是角色名, 第二个是模板字符串

    2. 消息对象写法 (需要精确控制时):
       SystemMessagePromptTemplate.from_template("你是{role}")

    3. 混合写法:
       可以把元组、对象、Placeholder 混合在一个列表里

    什么时候用哪种?
      - 元组写法: 90% 的场景, 够用了
      - 消息对象写法: 需要给消息加 extra_kwargs 或 metadata 时
    """
    print("\n" + "=" * 60)
    print("【第六部分】from_messages 的三种写法")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 写法 1: 元组 (推荐)
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "你是{role}"),
        ("user", "{input}"),
    ])
    result1 = model.invoke(prompt1.format_messages(role="医生", input="头疼怎么办?"))
    print(f"[元组写法] {result1.content[:80]}...")

    # 写法 2: MessagePromptTemplate 对象 (需要细粒度控制时)
    prompt2 = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("你是{role}, 专业领域: {domain}"),
        HumanMessagePromptTemplate.from_template("请分析: {input}"),
    ])
    result2 = model.invoke(prompt2.format_messages(
        role="医生", domain="神经内科", input="长期偏头痛的原因"
    ))
    print(f"[对象写法] {result2.content[:80]}...")

    # 写法 3: 混合
    prompt3 = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个通用助手"),
        ("user", "{input}"),
    ])
    result3 = model.invoke(prompt3.format_messages(input="Python 是哪种语言?"))
    print(f"[混合写法] {result3.content[:80]}...")


# ============================================================================
# 第七部分: Chain of Thought —— 让模型"先想后答"
# ============================================================================

def demo_7_cot():
    """
    Chain of Thought (思维链) 是什么?

    不是 LangChain 特有的功能, 而是一个 prompt 技巧:
    让模型在给出最终答案前, 先展示推理过程。

    效果: 显著提升复杂推理任务的准确率。

    方式:
      直接在 system prompt 里要求 "请逐步分析" 或 "先展示思路再给出答案"
      或者用 few-shot 示例展示期望的格式

    适用场景:
      - 数学计算
      - 逻辑推理
      - 代码分析
      - 多步骤决策
    """
    print("\n" + "=" * 60)
    print("【第七部分】Chain of Thought 提示技巧")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 无 CoT 的普通 prompt
    no_cot = ChatPromptTemplate.from_messages([
        ("system", "回答问题"),
        ("user", "小明有 5 个苹果, 给了小红 2 个, 又买了 3 个, 还剩几个?"),
    ])
    result1 = model.invoke(no_cot.format_messages())
    print(f"[无 CoT] {result1.content[:100]}...")

    # 有 CoT 的 prompt
    with_cot = ChatPromptTemplate.from_messages([
        ("system", "你是数学老师。请按以下格式回答:\n"
                   "【分析】逐步列出已知信息和计算步骤\n"
                   "【答案】给出最终结果"),
        ("user", "小明有 5 个苹果, 给了小红 2 个, 又买了 3 个, 还剩几个?"),
    ])
    result2 = model.invoke(with_cot.format_messages())
    print(f"\n[有 CoT] {result2.content[:200]}...")


# ============================================================================
# 第八部分: 实际项目中的 Prompt 组织
# ============================================================================

def demo_8_real_world():
    """
    在实际项目中, 不要把所有 prompt 散落在代码各处。
    推荐的组织方式:

    1. 集中管理: 放在一个 prompts.py 里
    2. 分层设计:
       - 基础层: 定义角色、风格、格式
       - 任务层: 定义具体任务 (翻译、总结、分类...)
       - 调用层: 组合基础层和任务层, 传入用户输入
    3. 版本管理: prompt 和代码一起用 git 管理
    4. 测试: 对关键 prompt 写单元测试, 验证格式正确

    示例:
      # prompts.py
      BASE_SYSTEM = "你是{role}, 回答要{style}。"
      TRANSLATE_TASK = "将以下文本翻译成{target_lang}: {text}"
      SUMMARY_TASK = "用{length}总结以下内容: {text}"

      # translate.py
      prompt = ChatPromptTemplate.from_messages([
          ("system", BASE_SYSTEM),
          ("user", TRANSLATE_TASK),
      ])
    """
    print("\n" + "=" * 60)
    print("【第八部分】实际项目中的 Prompt 组织")
    print("=" * 60)
    print("""
推荐的项目结构:
  prompts/
    __init__.py
    base.py          # 基础系统提示
    translate.py     # 翻译相关
    summarize.py     # 总结相关
    classify.py      # 分类相关

每个文件返回 ChatPromptTemplate 或 PromptTemplate 对象,
让调用方只关心传入参数, 不关心 prompt 的内部构造。
""")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_two_templates()
    demo_2_message_roles()
    demo_3_placeholder()
    demo_4_few_shot()
    demo_5_partial()
    demo_6_message_formats()
    demo_7_cot()
    demo_8_real_world()

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. PromptTemplate 生成字符串, ChatPromptTemplate 生成消息列表
2. system 设规则, user 是输入, assistant 是历史, 角色不能混用
3. MessagesPlaceholder 用于动态插入消息列表 (历史/检索结果)
4. FewShotPromptTemplate 用示例教会模型输出格式
5. Partial 提前绑定不变变量, 减少重复、防止遗漏
6. from_messages 支持元组、对象、混合三种写法
7. Chain of Thought 通过思维链提示提升推理准确率
8. 生产项目把 prompt 集中管理, 不要散落各处

下一节 03_output_parser.py 会深入如何让模型输出结构化数据。
""")
