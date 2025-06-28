from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Optional
from openai import OpenAI

from app.models.schemas import UploadResponse, ErrorResponse
from app.core.document_processor import DocumentProcessor
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
    description="基于 OpenAI 和向量检索的文档问答系统"
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

# 创建文档处理器实例
doc_processor = DocumentProcessor(UPLOAD_DIR, INDEX_DIR, OPENAI_API_KEY, OPENAI_API_BASE)

# 创建 OpenAI 客户端
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE  # 添加 base_url 配置
)

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
        if doc_processor.process_file(file_id, file_path):
            return UploadResponse(
                message="文件上传并处理成功",
                filename=file.filename,
                file_id=file_id
            )
        else:
            raise HTTPException(status_code=500, detail="文件处理失败")
            
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
        
        # 构建 prompt
        context = "\n\n".join([r["content"] for r in results])
        messages = [
            {"role": "system", "content": "你是一个专业的助手，请基于提供的上下文回答用户的问题。如果问题无法从上下文中得到答案，请说明。"},
            {"role": "user", "content": f"基于以下上下文回答问题：\n\n{context}\n\n问题：{question}"}
        ]
        
        # 调用 OpenAI 生成回答
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [{
                "content": r["content"],
                "file_id": r["metadata"]["file_id"],
                "score": r["score"]
            } for r in results]
        }
        
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