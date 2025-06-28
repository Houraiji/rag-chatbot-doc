import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-1106")

# 向量数据库配置
CHROMA_PERSIST_DIRECTORY = os.path.join(ROOT_DIR, os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/index"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

# Redis配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# 应用配置
UPLOAD_DIR = os.path.join(ROOT_DIR, os.getenv("UPLOAD_DIR", "./data/uploads"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", 5))
KEYWORD_SEARCH_TOP_K = int(os.getenv("KEYWORD_SEARCH_TOP_K", 3))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.7))  # 向量搜索权重，1-HYBRID_ALPHA为关键词搜索权重

# 确保必要的目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True) 