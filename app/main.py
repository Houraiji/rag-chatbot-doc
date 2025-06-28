from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.endpoints import router as api_router
from app.core.config import UPLOAD_DIR

# 创建 FastAPI 应用实例
app = FastAPI(
    title="RAG智能文档问答系统",
    description="基于LangChain的RAG系统，支持多轮对话记忆和混合检索策略",
    version="0.1.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保上传文件的目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 添加API路由
app.include_router(api_router) 