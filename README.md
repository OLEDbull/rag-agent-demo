# 智能扫地机器人客服 Agent 系统

基于 **LangChain + LangGraph** 构建的 **ReAct 模式智能客服 Agent**，专注于扫地机器人和扫拖一体机器人领域的专业问答。

## 核心能力

| 能力 | 说明 |
|------|------|
| **RAG 检索增强生成** | 从本地知识库检索相关文档，结合大模型生成专业回答 |
| **ReAct 自主推理** | Agent 自主判断是否需要调用工具、调用哪个工具、如何整合信息 |
| **多工具协同** | 集成知识检索、天气查询、用户定位、报告生成等 7 个工具 |
| **流式输出** | 回答逐字实时显示，配合 Spinner 加载动画提升体验 |
| **动态提示词切换** | 根据用户意图（普通问答 vs 报告生成）自动切换系统提示词 |
| **混合检索** | 向量语义检索 + 关键词精确匹配，双重保障检索质量 |

## 项目架构

```
用户提问 → Streamlit Web 界面
              │
              ▼
         ReactAgent (ReAct 循环)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  RAG检索   天气查询   报告生成  ... (7个工具)
    │
    ▼
  Chroma 向量库 (BGE-M3 嵌入)
    │
    ▼
  知识库文档 (data/*.txt, *.pdf)
```

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 配置

1. 在 `model/factory.py` 中配置 API Key：

```python
chat_model = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    api_key="你的API密钥",          # ← 替换为你的 SiliconFlow API Key
    base_url="https://api.siliconflow.cn/v1",
    timeout=60,
    max_retries=2,
)
```

2. 在 `config/rag.yml` 中确认模型配置：

```yaml
chat_model_name: Qwen/Qwen3-8B
embedding_model_name: BAAI/bge-m3
```

### 加载知识库

```bash
python rag/vector_store.py
```

### 启动服务

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

## 项目结构

```
d:\AI\agent\
├── app.py                    # Streamlit Web 入口
├── requirements.txt          # Python 依赖清单
│
├── agent/                    # Agent 智能体模块
│   ├── __init__.py           #   包初始化
│   ├── react_agent.py        #   ReAct Agent（创建、执行、流式输出）
│   └── tools/
│       ├── __init__.py       #     包初始化
│       ├── agent_tools.py    #     7 个工具定义
│       └── middleware.py     #     中间件（日志 + 提示词切换）
│
├── model/
│   ├── __init__.py           #   包初始化
│   └── factory.py            # ChatModel + EmbeddingModel 实例化
│
├── rag/                      # RAG 检索增强生成模块
│   ├── __init__.py           #   包初始化
│   ├── rag_service.py        #   检索 + LLM 总结服务
│   └── vector_store.py       #   Chroma 向量库服务
│
├── utils/                    # 通用工具模块
│   ├── __init__.py           #   包初始化
│   ├── config_handler.py     #   YAML 配置加载
│   ├── prompt_loader.py      #   提示词加载
│   ├── path_tool.py          #   路径工具
│   ├── logger_handler.py     #   日志系统
│   └── file_handler.py       #   文件处理（MD5、PDF/TXT 加载）
│
├── config/                   # YAML 配置文件
│   ├── rag.yml               #   模型名称配置
│   ├── chroma.yml            #   向量库参数
│   ├── prompts.yml           #   提示词路径
│   └── agent.yml             #   Agent 配置
│
├── prompts/prompts/          # 提示词模板
│   ├── main_prompt.txt       #   主系统提示词
│   ├── rag_summarize.txt     #   RAG 总结提示词
│   └── report_prompt.txt     #   报告生成提示词
│
├── data/                     # 知识库原始文档
├── chroma_db/                # Chroma 向量库持久化目录
└── logs/                     # 运行日志
```

## 可用工具

| 工具 | 功能 | 参数 |
|------|------|------|
| `rag_summarize` | RAG 知识检索 + LLM 总结 | `query: str` |
| `get_weather` | 天气查询 | `city: str` |
| `get_user_location` | 用户定位 | 无 |
| `get_user_id` | 获取用户 ID | 无 |
| `get_current_month` | 获取当前月份 | 无 |
| `fill_context_for_report` | 报告上下文注入 | 无 |
| `fetch_external_data` | 外部数据查询 | `user_id, month` |

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM 对话模型 | Qwen3-8B (SiliconFlow) |
| 嵌入模型 | BGE-M3 (SiliconFlow) |
| Agent 框架 | LangChain + LangGraph |
| 向量数据库 | Chroma |
| Web 界面 | Streamlit |
| 配置管理 | PyYAML |
| PDF 解析 | pypdf |

## 依赖说明

核心依赖（见 `requirements.txt`）：

| 包名 | 用途 |
|------|------|
| `langchain` | LLM 应用开发框架 |
| `langchain-openai` | OpenAI 兼容 API 接入（ChatOpenAI、OpenAIEmbeddings） |
| `langchain-chroma` | Chroma 向量库集成 |
| `langchain-text-splitters` | 文档切分（RecursiveCharacterTextSplitter） |
| `langchain-community` | 社区工具（PyPDFLoader、TextLoader） |
| `langgraph` | ReAct Agent 状态图引擎 |
| `langgraph-prebuilt` | 预构建工具节点（ToolNode） |
| `chromadb` | 向量数据库 |
| `streamlit` | Web 界面 |
| `pyyaml` | YAML 配置解析 |
| `pypdf` | PDF 文档解析 |

## 设计理念

系统核心思想是 **"让模型自己决定需要什么信息"**。与传统程序不同，调用哪个工具、以什么参数调用、调用几次，全部由大模型在运行时自主决定，而非开发者预先写死的逻辑。

```
用户提问 → Agent 思考 → 信息不足？→ 调用工具 → 获取结果 → 再思考 → 足够？→ 生成回答
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 一直显示"思考中"无输出 | 检查 API Key 是否有效、网络连接是否正常 |
| 向量库检索无结果 | 运行 `python rag/vector_store.py` 加载文档 |
| 工具调用报错 | 查看 `logs/agent_*.log` 日志文件 |

