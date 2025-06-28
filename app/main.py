from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Optional

from app.models.schemas import UploadResponse, ErrorResponse
from app.core.document_processor import DocumentProcessor

# 允许的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

# 创建上传文件的目录路径
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

# 创建 FastAPI 应用实例
app = FastAPI(
    title="智能文档问答系统",
    description="一个简单的文档问答系统"
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

# 创建文档处理器实例
doc_processor = DocumentProcessor(UPLOAD_DIR)

@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
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
        if doc_processor.process_file(file_id, file_path):
            return UploadResponse(
                message="文件上传并处理成功",
                filename=file.filename,
                file_id=file_id
            )
        else:
            raise HTTPException(status_code=500, detail="文件处理失败")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.post("/ask")
async def ask_question(question: str):
    """问答接口"""
    try:
        # 搜索相关文档
        results = doc_processor.search_documents(question)
        
        if not results:
            return {
                "question": question,
                "answer": "抱歉，我在文档中没有找到相关信息。",
                "sources": []
            }
        
        # 简单起见，直接返回找到的上下文
        return {
            "question": question,
            "answer": f"找到以下相关内容：\n\n{results[0]['context']}",
            "sources": [{"file_id": r["file_id"], "score": r["score"]} for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# 新增：获取已上传文件列表
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