from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Optional, Dict
from pydantic import BaseModel

from app.models.schemas import UploadResponse, ErrorResponse
from app.core.retriever import DocumentRetriever
from app.core.qa import QAChain
from app.core.config import (
    UPLOAD_DIR, 
    INDEX_DIR, 
    OPENAI_API_KEY, 
    OPENAI_API_BASE,
    OPENAI_MODEL,
    ALLOWED_EXTENSIONS
)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="智能文档问答系统",
    description="基于 LangChain 的文档问答系统，支持多轮对话"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保必要的目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# 创建文档检索器
retriever = DocumentRetriever(
    UPLOAD_DIR, 
    INDEX_DIR, 
    OPENAI_API_KEY,
    OPENAI_API_BASE
)

# 创建问答链
qa_chain = QAChain(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model_name=OPENAI_MODEL
)

# 定义请求和响应模型
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    sources: List[Dict] = []

class SessionResponse(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def root():
    """根路由"""
    return {"message": "欢迎使用智能文档问答系统！"}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """文件上传接口"""
    try:
        # 检查文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_ext}，目前仅支持 {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # 生成唯一文件ID和文件名
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 处理文件
        retriever.add_document(file_path, file_id)
        
        return UploadResponse(
            message="文件上传并处理成功",
            filename=file.filename,
            file_id=file_id
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """问答接口，支持多轮对话"""
    try:
        question = request.question
        session_id = request.session_id or str(uuid.uuid4())
        
        # 检索相关文档
        docs = await retriever.retrieve(question)
        
        if not docs:
            answer = "抱歉，我在文档中没有找到相关信息。"
            return QuestionResponse(
                question=question,
                answer=answer,
                session_id=session_id,
                sources=[]
            )
        
        # 使用 QA 链生成回答
        answer = await qa_chain.answer_question(
            question=question,
            documents=docs,
            session_id=session_id
        )
        
        return QuestionResponse(
            question=question,
            answer=answer,
            session_id=session_id,
            sources=[{
                "content": doc.page_content[:200] + "...",  # 截断显示
                "file_id": doc.metadata.get("file_id", "未知"),
                "source": doc.metadata.get("source", "未知")
            } for doc in docs]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/files")
async def list_files():
    """获取已上传的文件列表"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS:
                file_path = os.path.join(UPLOAD_DIR, filename)
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "upload_time": os.path.getctime(file_path)
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())
    return SessionResponse(
        session_id=session_id,
        message="会话创建成功"
    )

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史"""
    messages = qa_chain.get_message_history(session_id)
    return {"session_id": session_id, "messages": messages}

@app.delete("/sessions/{session_id}", response_model=SessionResponse)
async def delete_session(session_id: str):
    """删除会话"""
    if qa_chain.delete_session(session_id):
        return SessionResponse(
            session_id=session_id,
            message="会话删除成功"
        )
    raise HTTPException(status_code=404, detail="会话不存在")

@app.post("/sessions/{session_id}/clear", response_model=SessionResponse)
async def clear_session(session_id: str):
    """清除会话历史"""
    if qa_chain.clear_history(session_id):
        return SessionResponse(
            session_id=session_id,
            message="会话历史清除成功"
        )
    raise HTTPException(status_code=404, detail="会话不存在")