import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
async def root():
    """根路由，返回欢迎信息"""
    return {"message": "欢迎使用RAG问答系统~"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "version": "0.1.0"}

def main():
    """主程序入口"""
    print("启动RAG智能文档问答系统...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
