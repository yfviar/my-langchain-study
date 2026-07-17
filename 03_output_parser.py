"""
================================================================================
03 - 输出解析器 (OutputParser) 深度解析
================================================================================

你会学到:
  1. 为什么 LLM 的输出不能直接当数据用?
  2. StrOutputParser: 不止是取 .content, 还有流式处理
  3. JsonOutputParser: 内部如何处理错误的 JSON? (重试机制)
  4. PydanticOutputParser: 自动生成 format_instructions 的原理
  5. format_instructions 怎么工作? 为什么能约束模型输出?
  6. 容错处理: 解析失败怎么办?
  7. 自定义 Parser: 什么时候需要自己写一个?

================================================================================
核心问题: 为什么需要 OutputParser?

LLM 返回的是自然语言文本, 不可靠:
  - 同一个问题, 回答格式每次都不同
  - 有时带解释, 有时候只有结论
  - JSON 可能有语法错误 (多余逗号, 缺引号)

OutputParser 的作用:
  1. 标准化输出: 把 AIMessage 转换成代码可以消费的数据
  2. 容错: 处理格式不完美的输出
  3. 类型安全: PydanticOutputParser 保证输出符合 schema

它在 chain 里的位置永远是最后一步:
  prompt | model | parser
                   ^^^^^^
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    BaseOutputParser,
)
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# 第一部分: StrOutputParser 的内部机制
# ============================================================================

def demo_1_str_parser():
    """
    StrOutputParser 看起来很简单, 但它做了三件事:

    1. 对 AIMessage: 提取 .content 属性
    2. 对字符串:    直接返回 (幂等, 可重复调用)
    3. 对流式 chunk: 提取 .content, 支持 stream 模式

    你可以用 parser.invoke(aimessage) 或 parser.invoke("text") 验证。
    这就是为什么它叫 "串的" 而不是 "精确的" ——
    它能处理多种输入类型, 保证输出是 str。
    """
    print("=" * 60)
    print("【第一部分】StrOutputParser 内部机制")
    print("=" * 60)

    parser = StrOutputParser()

    # 输入是 AIMessage
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="这是一段回复")
    result1 = parser.invoke(msg)
    print(f"[输入: AIMessage] 类型: {type(result1).__name__}, 内容: {result1}")

    # 输入是普通字符串 (幂等)
    result2 = parser.invoke("直接是字符串")
    print(f"[输入: str]       类型: {type(result2).__name__}, 内容: {result2}")

    # 流式模式
    print("[流式模式] ", end="", flush=True)
    for chunk in parser.stream(msg):
        print(chunk, end="", flush=True)
    print()


# ============================================================================
# 第二部分: JsonOutputParser —— 让模型输出 JSON
# ============================================================================

def demo_2_json_parser():
    """
    JsonOutputParser 做了什么？

    1. 在 prompt 里加入 "请返回 JSON 格式" 的指令
       但这不是它加入的, 需要你自己在 prompt 里写

    2. 把模型的文本输出传给 json.loads()
       如果解析成功 → 返回 dict
       如果解析失败 → 抛出 OutputParserException

    3. (重要!) 它有内置的重试机制吗？
       没有! 如果 JSON 解析失败, 直接报错。
       你需要在外面捕获异常, 或者用 OutputFixingParser。

    核心痛点:
      模型生成的 JSON 经常有问题:
        - 多了一个尾部逗号: {"a": 1,}
        - 少了一个引号:   {"a": 1
        - 在 JSON 前后加了说明文字: "好的, 这是结果: {...}"
        - 用中文引号:       {"a": "你好"}
        - 用了 undefined:   {"a": undefined}

    解决方案:
      1. 在 prompt 里强调 "只返回 JSON, 不要其他内容"
      2. 用 PydanticOutputParser 代替 (有 schema 约束, 更可靠)
      3. 用 json_repair 库预处理输出
      4. 加 retry 逻辑
    """
    print("\n" + "=" * 60)
    print("【第二部分】JsonOutputParser")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 好的 prompt: 明确要求"只返回 JSON"
    prompt = ChatPromptTemplate.from_template(
        """从以下文本提取人物信息, 只返回 JSON 对象, 不要其他内容。
字段: name (姓名), age (年龄, 数字), city (城市)

文本: {text}"""
    )
    chain = prompt | model | JsonOutputParser()

    result = chain.invoke({"text": "王小明今年 28 岁, 住在上海浦东。"})
    print(f"[成功解析] 类型: {type(result).__name__}")
    print(f"  name: {result.get('name')}")
    print(f"  age:  {result.get('age')} (类型: {type(result.get('age')).__name__})")
    print(f"  city: {result.get('city')}")

    # 容错演示: 如果 prompt 写得太含糊
    bad_prompt = ChatPromptTemplate.from_template("{text}")
    bad_chain = bad_prompt | model | JsonOutputParser()

    try:
        bad_chain.invoke({"text": "把王小明28岁的信息用JSON返回给我"})
    except OutputParserException as e:
        print(f"\n[解析失败] {str(e)[:100]}...")
        print("[说明] 模型可能返回了非 JSON 内容, 这就是不用 PydanticParser 的风险")


# ============================================================================
# 第三部分: PydanticOutputParser —— 类型安全的解析
# ============================================================================

# 定义我们期望的数据结构 (生产代码应该放在 models.py 里)
class MovieReview(BaseModel):
    """影评数据结构"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分, 1.0-10.0")
    summary: str = Field(description="一句话评价, 不超过50字")
    pros: List[str] = Field(description="优点, 至少1条")
    cons: List[str] = Field(description="缺点, 至少1条")
    recommended: bool = Field(description="是否推荐观看")


class PersonInfo(BaseModel):
    """个人信息数据结构"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    skills: List[str] = Field(description="技能列表")


def demo_3_pydantic_parser():
    """
    PydanticOutputParser 的工作原理:

    1. 你定义 Pydantic Model (就是 Python 的 dataclass Puls)
    2. 把 Model 传给 PydanticOutputParser
    3. parser.get_format_instructions() 自动生成一段 "输出格式要求"
       这段文本会被拼到 prompt 里, 告诉模型该输出什么 JSON 结构

    format_instructions 长什么样?

    比如 MovieReview 生成的指令大约是这样的:
      The output should be formatted as a JSON instance that conforms to
      the JSON schema below.
      {
        "properties": {
          "title": {"description": "电影名称", "type": "string"},
          "rating": {"description": "评分, 1.0-10.0", "type": "number"},
          ...
        },
        "required": ["title", "rating", "summary", "pros", "cons", "recommended"]
      }

    为什么比 JsonOutputParser 更好?
      1. schema 自动生成, 不需要手写 "请返回 name, age, city" 这样的提示
      2. 类型安全: 模型返回后自动转成 Pydantic 对象
      3. 验证: 如果模型返回的字段类型不对, Pydantic 会报错

    但是也有局限:
      - 模型不一定严格遵守 schema (特别是小模型)
      - 复杂嵌套结构解析成功率下降
    """
    print("\n" + "=" * 60)
    print("【第三部分】PydanticOutputParser —— 类型安全解析")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    # 查看自动生成的格式指令
    instructions = parser.get_format_instructions()
    print(f"[自动生成的 format_instructions]")
    print(instructions[:300], "...\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业影评人。\n{format_instructions}"),
        ("user", "请评价电影: {movie}"),
    ]).partial(format_instructions=instructions)

    chain = prompt | model | parser
    result = chain.invoke({"movie": "流浪地球"})

    print(f"[解析结果]")
    print(f"  类型: {type(result).__name__}")
    print(f"  电影: {result.title}")
    print(f"  评分: {result.rating}")
    print(f"  评价: {result.summary}")
    print(f"  推荐: {result.recommended}")
    print(f"  优点: {result.pros}")
    print(f"  缺点: {result.cons}")


# ============================================================================
# 第四部分: 解析失败 → 怎么容错
# ============================================================================

def demo_4_error_handling():
    """
    解析失败的几种处理策略:

    策略 1: 重试 (RetryWithErrorOutputParser)
      把解析错误信息反馈给模型, 让它修正

    策略 2: 修复 (json_repair)
      用第三方库尝试修复有问题的 JSON

    策略 3: 兜底 (OutputFixingParser)
      用另一个模型来修正输出

    策略 4: 降级
      解析不了就算了, 返回原始文本

    实际项目中, 常见做法:
      1. prompt 里反复强调 "只返回 JSON"
      2. 如果解析失败, 用正则提取可能的 JSON 部分
      3. 再失败, 重试一次
      4. 最终还是失败, 记录原始输出, 人工介入
    """
    print("\n" + "=" * 60)
    print("【第四部分】解析失败的容错策略")
    print("=" * 60)

    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    instructions = parser.get_format_instructions()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{format_instructions}\n只返回 JSON, 不要其他文字!"),
        ("user", "描述这个人: {description}"),
    ]).partial(format_instructions=instructions)

    chain = prompt | model | parser

    try:
        result = chain.invoke({"description": "张三会写 Python 和 Java, 今年 30 岁"})
        print(f"[成功] {result.name}, {result.age}岁, 技能: {result.skills}")
    except OutputParserException as e:
        print(f"[失败] {e}")

    # 模拟容错: catch 异常后尝试修复
    print("""
容错的最佳实践:

    try:
        result = chain.invoke(input)
    except OutputParserException:
        # 1. 提取可能的 JSON 部分
        text = raw_response.content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            # 2. 尝试 json.loads
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                # 3. 如果还是失败, 降级处理
                data = fallback_parse(text)
    """)


# ============================================================================
# 第五部分: 自定义 OutputParser
# ============================================================================

def demo_5_custom_parser():
    """
    什么时候需要自定义 Parser?

    - 输出格式不是 JSON (如 CSV, YAML, Markdown 表格)
    - 需要特定的后处理 (如去空格、转小写、提取第一行)
    - 需要调用外部服务验证输出

    自定义 Parser 只需继承 BaseOutputParser, 实现两个方法:
      parse(text: str) -> T
      get_format_instructions() -> str

    但大部分场景可以用 BaseTransformOutputParser:
      transform(text: str) -> str
      在 parse 之前对文本做预处理
    """
    print("\n" + "=" * 60)
    print("【第五部分】自定义 OutputParser")
    print("=" * 60)

    from langchain_core.output_parsers import BaseOutputParser

    class TrimAndLowerParser(BaseOutputParser[str]):
        """去除首尾空白并转小写"""

        def parse(self, text: str) -> str:
            return text.strip().lower()

        def get_format_instructions(self) -> str:
            return "返回纯文本, 输出会被自动格式化。"

        @property
        def _type(self) -> str:
            return "trim_and_lower"

    parser = TrimAndLowerParser()
    result = parser.parse("  HELLO World!  \n")
    print(f"[自定义 Parser] 输入: '  HELLO World!  \\n'")
    print(f"[自定义 Parser] 输出: '{result}'")


# ============================================================================
# 第六部分: 链式 Parser —— 多个 Parser 组合
# ============================================================================

def demo_6_chained_parsers():
    """
    多个 Parser 可以串联:

    model | parse_json | extract_field | validate

    每个 Parser 做一件事, 像 Unix 管道一样。
    这也是 | 运算符的威力 —— Parser 也是 Runnable 的一种。

    场景: 先解析 JSON, 再提取特定字段, 再验证合法性
    """
    print("\n" + "=" * 60)
    print("【第六部分】链式 Parser")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_template(
        """返回 JSON 格式: {{"name": "姓名", "email": "邮箱", "phone": "电话"}}
从文本提取: {text}"""
    )

    from langchain_core.runnables import RunnableLambda

    def extract_and_validate(data: dict) -> str:
        email = data.get("email", "")
        import re
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            raise OutputParserException(f"无效的邮箱格式: {email}")
        return email

    chain = prompt | model | JsonOutputParser() | RunnableLambda(extract_and_validate)
    result = chain.invoke({"text": "联系我: 张三, zhang@company.com, 13800138000"})
    print(f"[Parser 链] 提取的邮箱: {result}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_str_parser()
    demo_2_json_parser()
    demo_3_pydantic_parser()
    demo_4_error_handling()
    demo_5_custom_parser()
    demo_6_chained_parsers()

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. OutputParser 解决 LLM 输出不可靠的问题, 标准化下游消费
2. StrOutputParser 处理 AIMessage/str/流式三种输入
3. JsonOutputParser 只是 json.loads(), 没有内置容错
4. PydanticOutputParser 自动生成 schema, 提供类型安全保障
5. get_format_instructions 把 schema 转成模型能理解的指令
6. 容错策略: prompt 强调 -> 异常捕获 -> json_repair -> 降级
7. 自定义 Parser 只需实现 parse() 和 get_format_instructions()
8. Parser 也是 Runnable, 可以用 | 串联多个

下一节 04_rag.py 会深入 RAG 检索增强生成的全流程。
""")
