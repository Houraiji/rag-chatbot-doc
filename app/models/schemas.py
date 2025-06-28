from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., description="用户提问的问题")
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话，如果为空则创建新会话")


class DocumentChunk(BaseModel):
    """文档块模型"""
    content: str = Field(..., description="文档块内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档块元数据")
    score: Optional[float] = Field(None, description="相关性分数")


class QuestionResponse(BaseModel):
    """问题回答响应模型"""
    question: str = Field(..., description="用户提问的问题")
    answer: str = Field(..., description="系统回答")
    session_id: str = Field(..., description="会话ID")
    sources: Optional[List[DocumentChunk]] = Field(None, description="回答的来源文档块")
    processing_time: float = Field(..., description="处理时间(秒)")


class UploadResponse(BaseModel):
    """文件上传响应模型"""
    message: str = Field(..., description="上传结果消息")
    filename: str = Field(..., description="上传的文件名")
    file_id: str = Field(..., description="文件唯一标识符")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    details: Optional[str] = Field(None, description="错误详情") 