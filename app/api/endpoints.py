import os
import time
import shutil
from typing import List, Dict, Any, Optional
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import QuestionRequest, QuestionResponse, UploadResponse, ErrorResponse
from app.core.qa import DocumentQA
from app.core.config import UPLOAD_DIR


# 创建API路由
router = APIRouter()

# 创建DocumentQA实例
qa_pipeline = DocumentQA()


@router.get("/")
async def root():
    """根路由"""
    return {"message": "欢迎使用RAG问答系统~"}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """上传文件API
    
    Args:
        file: 上传的文件
        background_tasks: 后台任务
        
    Returns:
        上传结果
    """
    try:
        # 检查文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".pdf", ".txt", ".md", ".csv"]:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_ext}，目前仅支持PDF、TXT、MD和CSV文件"
            )
        
        # 生成唯一文件ID
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 确保上传目录存在
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 在后台处理文件
        background_tasks.add_task(qa_pipeline.load_and_index_pdf, file_path)
        
        return {
            "message": "文件上传成功，正在后台处理",
            "filename": file.filename,
            "file_id": file_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """问答API
    
    Args:
        request: 问题请求
        
    Returns:
        问题回答
    """
    try:
        # 检查问题是否为空
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # 回答问题
        result = qa_pipeline.answer_question(
            question=request.question,
            session_id=request.session_id
        )
        
        return {
            "question": request.question,
            "answer": result["answer"],
            "session_id": result["session_id"],
            "sources": result.get("sources"),
            "processing_time": result["processing_time"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-index/")
async def test_index():
    """测试索引API"""
    try:
        index_loaded = qa_pipeline.load_existing_index()
        if index_loaded:
            return {"status": "success", "message": "索引加载成功"}
        else:
            return {"status": "error", "message": f"索引加载失败，路径 '{qa_pipeline.index_path}' 不存在或为空"}
    except Exception as e:
        return {"status": "error", "message": f"加载索引时出错: {str(e)}"}


@router.get("/sessions/")
async def list_sessions():
    """列出所有会话"""
    try:
        sessions = qa_pipeline.memory_manager.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        if not qa_pipeline.memory_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        qa_pipeline.memory_manager.delete_session(session_id)
        return {"status": "success", "message": f"会话 {session_id} 已删除"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))