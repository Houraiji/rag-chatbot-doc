# RAG-SmartChat

RAG-SmartChat 是一个基于 LangChain 0.3 构建的检索增强生成（Retrieval-Augmented Generation, RAG）智能文档问答系统，支持多轮对话记忆和混合检索策略。

## 项目简介

该系统允许用户上传文档，然后针对这些文档进行提问。系统使用 LangChain 的向量检索技术找到最相关的文档片段，并结合大语言模型生成准确的回答。

### 主要特点

- **基于 LangChain 0.3**：使用最新的 LangChain API 和组件构建
- **文档处理**：支持 PDF、TXT、MD、CSV 等多种文件格式
- **向量检索**：使用 LangChain 的检索器接口和 Chroma 向量数据库
- **智能问答**：基于 LCEL（LangChain Expression Language）构建的 RAG 链
- **多轮对话**：使用 `RunnableWithMessageHistory` 实现对话记忆
- **会话管理**：创建、查询、清除和删除会话

## 技术栈

- **后端框架**：FastAPI
- **LangChain**：使用 LangChain 0.3+ 及其组件
  - `langchain_core`：核心组件和接口
  - `langchain_chroma`：向量存储
  - `langchain_openai`：OpenAI 集成
  - `langchain_community`：文档加载器
- **大语言模型**：OpenAI API
- **文档加载**：LangChain 文档加载器
- **向量嵌入**：OpenAI Embeddings

## 安装与配置

### 环境要求

- Python 3.10+
- 虚拟环境（推荐）

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/Houraiji/rag-smart-chat.git
cd rag-smart-chat
```

2. 创建并激活虚拟环境
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. 安装依赖
```bash
pip install -e .
```

4. 创建环境变量文件
```bash
cp .env.example .env
```

5. 编辑 `.env` 文件，设置 OpenAI API 密钥和其他配置

### 目录结构

```
rag-smartchat/
├── app/ # 应用代码
│ ├── api/ # API 接口
│ ├── core/ # 核心功能
│ │ ├── config.py # 配置
│ │ ├── retriever.py # 文档检索器
│ │ └── qa.py # 问答链
│ ├── models/ # 数据模型
│ └── main.py # 主程序入口
├── data/ # 数据目录
│ ├── index/ # 向量索引
│ └── uploads/ # 上传文件
├── tests/ # 测试代码
├── .env # 环境变量
├── .env.example # 环境变量示例
├── pyproject.toml # 项目配置
├── setup.py # 安装配置
└── README.md # 项目说明
```

## 使用方法

### 启动服务

```bash
cd rag-smartchat
uvicorn app.main:app --reload
```

服务将在 http://127.0.0.1:8000 启动，API 文档可访问 http://127.0.0.1:8000/docs