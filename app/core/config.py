import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# 文件配置
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

# 向量存储配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200