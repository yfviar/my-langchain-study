"""
================================================================================
04 - RAG (检索增强生成) 深度解析
================================================================================

你会学到:
  1. RAG 解决了什么问题? 为什么不能直接用 LLM?
  2. 文档加载: 不同的文件格式怎么处理?
  3. 文本分割: chunk_size 和 overlap 为什么重要?
  4. 向量化: Embedding 到底是什么? 怎么选?
  5. 向量存储: Chroma 的索引和检索原理
  6. 检索器: 如何控制检索质量?
  7. RAG Chain 的构造: RunnablePassthrough 的作用
  8. RAG 的局限和进阶方向

================================================================================
RAG 要解决的核心问题

LLM 的三大局限:
  1. 知识截止: 训练数据有截止日期, 不知道最新信息
  2. 幻觉: 不知道时会编造答案 (而且编得很像真的)
  3. 私有知识: 不知道你公司的内部文档

RAG 的思路: 检索 + 生成
  用户问问题 → 从知识库检索相关文档 → 把文档和问题一起给 LLM
  → LLM 基于文档内容回答

这就变成了 "带着参考资料回答问题", 而不是 "凭记忆回答"。
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()


# ============================================================================
# 第一部分: 为什么要 RAG? —— 无 RAG 的 LLM 有多糟糕
# ============================================================================

def demo_1_no_rag():
    """
    一个典型的 "无 RAG" 调用:
      LLM 完全凭训练数据回答, 不知道任何个性化信息。

    测试:
      问一个只有我们才知道的问题 (关于假想的 "ACME 公司")
      看看模型是诚实说不知道, 还是编造答案。
    """
    print("=" * 60)
    print("【第一部分】无 RAG —— LLM 的局限")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    questions = [
        "ACME 公司成立于哪一年? CEO 是谁?",
        "ACME 公司的主要产品是什么?",
        "ACME 公司的员工餐厅几点营业?",
    ]

    for q in questions:
        response = model.invoke(q)
        print(f"\n[问题] {q}")
        print(f"[回答] {response.content[:150]}...")

    print("\n[总结] 模型要么编造、要么说不知道。这就是需要 RAG 的原因。")


# ============================================================================
# 第二部分: 准备知识库文档
# ============================================================================

def create_sample_knowledge():
    """创建一份模拟的公司知识库文档"""
    content = """
# ACME 科技有限公司内部知识库

## 公司概况
ACME 科技有限公司成立于 2018 年 3 月, 总部位于杭州未来科技城。
创始人兼 CEO 是陈建国, CTO 是李敏。

## 主要业务
公司专注于企业级 AI 解决方案, 主要产品包括:
- ACME Chat: 企业智能客服系统, 支持多轮对话和知识库管理
- ACME Doc: 智能文档处理平台, 支持 OCR、分类、信息提取
- ACME Data: 数据分析助手, 支持自然语言查询数据库

## 公司制度
- 工作时间: 9:00 - 18:00, 弹性打卡
- 员工餐厅: 午市 11:30-13:00, 晚市 17:30-19:00
- 健身房: 24 小时开放, 位于 B1 层

## 技术栈
后端: Python, FastAPI, PostgreSQL, Redis
前端: React, TypeScript, Ant Design
AI: LangChain, OpenAI, 自研模型 ACME-LM-7B
基础设施: Docker, Kubernetes, 阿里云

## 联系方式
官网: https://acme-tech.com
招聘: hr@acme-tech.com
商务: sales@acme-tech.com
客服热线: 400-888-9999
"""
    doc_path = "acme_knowledge.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(content)
    return doc_path


# ============================================================================
# 第三部分: 文档加载 —— TextLoader 和其他 Loader
# ============================================================================

def demo_3_loading():
    """
    文档加载器 (Document Loader)

    TextLoader:     加载 .txt 文件
    PyPDFLoader:    加载 PDF
    CSVLoader:      加载 CSV
    UnstructuredMarkdownLoader: 加载 Markdown
    WebBaseLoader:  加载网页内容
    DirectoryLoader:加载整个目录

    每个 Loader 都返回 List[Document]。
    Document 是一个简单的容器:
      - page_content: str   (文档文本内容)
      - metadata: dict      (来源、页数、文件名等元信息)

    为什么需要 metadata?
      检索时需要知道每个 chunk 来自哪个文档、第几页,
      方便引用来源和追溯。
    """
    print("\n" + "=" * 60)
    print("【第三部分】文档加载")
    print("=" * 60)

    doc_path = create_sample_knowledge()
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    print(f"加载文档数: {len(documents)}")
    print(f"文档元信息: {documents[0].metadata}")
    print(f"文档长度: {len(documents[0].page_content)} 字符")

    return doc_path, documents


# ============================================================================
# 第四部分: 文本分割 —— chunk_size 和 overlap 的艺术
# ============================================================================

def demo_4_splitting(documents):
    """
    为什么需要分割?

    LLM 的上下文窗口有限 (4K, 8K, 128K tokens)。
    你不能把整本书塞进去。
    而且检索时, 小块更精准。

    两个关键参数:

    chunk_size (块大小):
      太大: 包含太多无关信息, 噪声大
      太小: 信息不完整, 一句话被截断
      推荐: 200-1000 (取决于文档类型和模型窗口)

    chunk_overlap (重叠量):
      为什么需要重叠?
      因为分割可能在句子中间切断。
      重叠让相邻 chunk 共享一部分内容, 保证信息的连续性。

      太大: 冗余多, 向量库体积大
      太小: 相邻 chunk 的信息断裂
      推荐: chunk_size 的 10%-20%

    RecursiveCharacterTextSplitter 的工作原理:
      按优先级尝试分割符:
        "\\n\\n" -> "\\n" -> "。" -> "." -> " " -> ""
      这样保证优先在段落和句子边界分割, 而不是在词中间切断。
    """
    print("\n" + "=" * 60)
    print("【第四部分】文本分割 —— chunk_size 和 overlap")
    print("=" * 60)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        # 分割优先级: 先按双换行, 再按单换行, 最后按字符
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    print(f"原始文档: 1 篇, {len(documents[0].page_content)} 字符")
    print(f"分割后: {len(chunks)} 个 chunk")
    print(f"平均大小: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")

    # 展示 overlap 的效果: 相邻两个 chunk 的公共部分
    if len(chunks) >= 2:
        print(f"\n[Chunk 1 结尾] ...{chunks[0].page_content[-80:]}")
        print(f"[Chunk 2 开头] {chunks[1].page_content[:80]}...")
        print(f"[说明] overlap 让相邻 chunk 有重叠, 保证信息连续性")

    return chunks


# ============================================================================
# 第五部分: 向量化和向量存储 —— Embedding 的原理
# ============================================================================

def demo_5_embedding(chunks):
    """
    Embedding 到底是什么?

    把一段文本变成一个固定长度的数字向量 (比如 1536 个浮点数)。
    语义相近的文本, 它们的向量在空间中就很接近。

    打个比方:
      "今天天气真好"    →  [0.1, 0.3, -0.2, ...]  (1536 个数)
      "今天的天气不错"  →  [0.11, 0.29, -0.19, ...] (几乎相同)
      "明天会下雨吗"    →  [-0.5, 0.1, 0.8, ...]   (完全不同)

    向量数据库 (Chroma) 做的事:
      1. 把所有 chunk 的向量存起来
      2. 用户提问题时, 把问题也向量化
      3. 在向量空间里找到最近的 K 个 chunk (余弦相似度)
      4. 返回这些 chunk 的原始文本

    Chroma  vs  其他向量数据库:
      Chroma:    轻量级, Python 原生, 适合开发和原型
      Milvus:    高性能, 分布式, 适合生产
      Pinecone:  云服务, 免运维
      FAISS:     Meta 出品, 纯本地, 极快
      Qdrant:    Rust 实现, 性能优秀

    选型建议:
      - 学习和小项目: Chroma
      - 生产环境: Milvus 或 Pinecone
    """
    print("\n" + "=" * 60)
    print("【第五部分】向量化和存储")
    print("=" * 60)

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=embedding_model,
    )
    print(f"[embedding 模型] {embedding_model}")

    # 查看一个文本的向量
    sample_text = "ACME 科技公司"
    sample_vector = embeddings.embed_query(sample_text)
    print(f"'{sample_text}' 的向量:")
    print(f"  维度: {len(sample_vector)}")
    print(f"  前 5 个值: {sample_vector[:5]}")
    print()

    # 存入 Chroma (persist 到磁盘)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print(f"向量库存储完成: {vectorstore._collection.count()} 条记录")
    print(f"持久化路径: ./chroma_db")

    return vectorstore


# ============================================================================
# 第六部分: 检索器 —— 如何控制检索质量
# ============================================================================

def demo_6_retrieval(vectorstore):
    """
    检索器的配置直接影响答案质量。

    search_type 的两种模式:
      similarity:  纯向量相似度 (默认)
      mmr:         最大边际相关性 (去重, 保证多样性)

    search_kwargs 的关键参数:
      k:              返回几条结果
      score_threshold: 最低相似度阈值 (低于此值的不返回)
      fetch_k:        MMR 时先取这么多再精选 (通常 k * 4)

    检索结果质量怎么评估?
      看 chunk 的内容和问题的语义相关性。
      如果检索到的都是无关内容, RAG 就失败了。

    常见问题:
      1. 检索不到相关内容 → chunk 太小或太大, 或 embedding 模型不合适
      2. 检索到太多不相关 → 需要 score_threshold 过滤
      3. 检索内容重复 → 用 MMR 模式去重
    """
    print("\n" + "=" * 60)
    print("【第六部分】检索器")
    print("=" * 60)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    queries = [
        "ACME 的工作时间是什么?",
        "公司有哪些 AI 产品?",
        "创始人是谁?",
    ]

    for q in queries:
        docs = retriever.invoke(q)
        print(f"\n[查询] {q}")
        for i, doc in enumerate(docs):
            print(f"  [结果 {i+1}] {doc.page_content[:100]}...")

    return retriever


# ============================================================================
# 第七部分: 组装 RAG Chain —— RunnablePassthrough 的作用
# ============================================================================

def demo_7_rag_chain(retriever):
    """
    RAG Chain 的构造:

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    拆解每一步:

    1. {"context": ..., "question": ...}
       这是一个 RunnableParallel, 同时执行两件事:
         context:  运行 retriever → 得到 List[Document] → format_docs 转成字符串
         question: RunnablePassthrough() 直接透传用户的输入

    2. | prompt
       把 {"context": "文档内容...", "question": "用户问题"} 填入模板

    3. | model
       LLM 基于上下文回答问题

    4. | parser
       提取文本

    RunnablePassthrough 的作用:
      就是 "什么都不做, 直接传过去"。
      因为 chain.invoke("问题") 传入的是一个字符串,
      而 prompt 需要 {"context": ..., "question": ...} 这样的 dict。
      上面那步 RunnableParallel 同时获取 context (从检索引擎) 和
      question (直接透传), 组合成完整的 dict。
    """
    print("\n" + "=" * 60)
    print("【第七部分】组装 RAG Chain")
    print("=" * 60)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = ChatPromptTemplate.from_template(
        """根据以下参考资料回答问题。如果资料中没有相关信息,
请明确说"资料中未找到相关信息", 不要编造。

参考资料:
{context}

问题: {question}

回答:"""
    )

    def format_docs(docs):
        """把检索到的文档列表拼接成一个字符串"""
        return "\n\n---\n\n".join(
            f"[来源: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    test_questions = [
        "ACME 公司的员工餐厅几点营业?",
        "公司技术栈里用了哪些前端技术?",
        "ACME 公司的 CEO 是谁?",
        "ACME 公司的股价是多少?",  # 知识库里没有
    ]

    for q in test_questions:
        result = rag_chain.invoke(q)
        print(f"\n[Q] {q}")
        print(f"[A] {result[:200]}...")

    return rag_chain


# ============================================================================
# 第八部分: RAG 的局限和进阶方向
# ============================================================================

def demo_8_limitations():
    """
    当前这个简单 RAG 还有很多问题:

    1. 检索不准
       用户问法多变, "CEO" 可能表述为 "一把手"、"老板"、"话事人"
       单纯向量检索找不到
       解决: 查询改写 (Query Rewriting), 混合检索 (关键词 + 向量)

    2. 检索到的内容不完整
       一个 chunk 可能只包含答案的一部分
       解决: Parent-Child chunking, 检索小块, 返回大块

    3. 幻觉依然存在
       即使给了参考资料, 模型仍可能编造
       解决: 更严格的 prompt, 加验证步骤

    4. 速度慢
       向量检索本身很快, 但 embedding 调用需要时间
       解决: 缓存 embedding, 用更快的模型

    5. 更新困难
       文档更新了, 需要重新 embedding
       解决: 增量索引, 版本管理

    进阶 RAG 技术栈:
      - 混合检索: 关键词检索 (BM25) + 向量检索
      - 重排序 (Rerank): 用更强的模型对检索结果重新排序
      - 查询分解: 复杂问题拆成子问题
      - 自查询 (Self-query): 让 LLM 自动把问题转成检索条件
    """
    print("\n" + "=" * 60)
    print("【第八部分】RAG 的局限和进阶方向")
    print("=" * 60)
    print("""
进阶 RAG 路线图:

  Level 1 (本节):  简单检索 → 生成
  Level 2:         查询改写 + 混合检索
  Level 3:         Rerank 重排序
  Level 4:         自查询 + 多步检索
  Level 5:         Agentic RAG (Agent 决定何时检索、检索什么)

每个 Level 都解决上一级的一个痛点。
""")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo_1_no_rag()
    doc_path, documents = demo_3_loading()
    chunks = demo_4_splitting(documents)
    vectorstore = demo_5_embedding(chunks)
    retriever = demo_6_retrieval(vectorstore)
    demo_7_rag_chain(retriever)
    demo_8_limitations()
    os.remove(doc_path)
    print(f"\n已清理临时文档: {doc_path}")
    print(f"向量库保留在 ./chroma_db, 下次运行会复用")

    print("\n" + "=" * 60)
    print("""
【学完这节你应该能回答】

1. RAG = 检索 + 生成, 解决 LLM 不知道私有知识和幻觉问题
2. Document Loader 把各种格式变成统一的 Document 对象
3. chunk_size 决定检索粒度, overlap 保证信息连续性
4. Embedding 把文本映射为向量, 语义相近 = 向量相近
5. 向量库存储索引, 用余弦相似度做近似搜索
6. retriever 的 k、score_threshold、MMR 控制检索质量
7. RunnablePassthrough 透传输入, RunnableParallel 并行执行
8. 简单 RAG 有检索不准、幻觉、更新困难等问题, 需要进阶技术

下一节 05_agent.py 会深入 Agent 如何自主决策和调用工具。
""")
