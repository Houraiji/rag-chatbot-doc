# RAG-SmartChat

RAG-SmartChat 是一个基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能文档问答系统，支持多轮对话和混合检索策略。

## 项目简介

该系统允许用户上传文档（如 PDF、TXT 等），然后针对这些文档进行提问。系统会使用向量检索技术找到最相关的文档片段，并结合大语言模型生成准确的回答。

### 主要特点

- **文档处理**：支持 PDF、TXT、MD、CSV 等多种文件格式
- **向量检索**：使用 Chroma 向量数据库存储和检索文档片段
- **智能问答**：基于 OpenAI 模型生成高质量回答
- **多轮对话**：支持上下文相关的多轮对话
- **会话管理**：创建、查询、清除和删除会话

## 技术栈

- **后端框架**：FastAPI
- **向量存储**：Chroma
- **大语言模型**：OpenAI API
- **文档处理**：LangChain、PyPDF
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
│ ├── models/ # 数据模型
│ ├── utils/ # 工具函数
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