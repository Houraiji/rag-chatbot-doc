import os
import uuid
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR


class DocumentProcessor:
    """文档处理器，用于解析和分块文档"""
    
    def __init__(self):
        """初始化文档处理器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        """处理文件，返回文档块列表
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档块列表
        """
        # 根据文件扩展名选择不同的加载器
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".pdf":
                return self._process_pdf(file_path)
            elif file_ext in [".txt", ".md", ".csv"]:
                return self._process_text(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
        except Exception as e:
            raise Exception(f"处理文件 {file_path} 时出错: {str(e)}")
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """处理PDF文件"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 添加元数据
        file_name = os.path.basename(file_path)
        file_id = str(uuid.uuid4())
        
        for doc in documents:
            doc.metadata.update({
                "source": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "pdf"
            })
        
        # 分块
        return self.text_splitter.split_documents(documents)
    
    def _process_text(self, file_path: str) -> List[Document]:
        """处理文本文件"""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # 添加元数据
        file_name = os.path.basename(file_path)
        file_id = str(uuid.uuid4())
        file_type = os.path.splitext(file_path)[1].lower().replace(".", "")
        
        for doc in documents:
            doc.metadata.update({
                "source": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": file_type
            })
        
        # 分块
        return self.text_splitter.split_documents(documents)
    
    def get_file_id(self, file_path: str) -> str:
        """生成文件ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str) -> str:
        """保存上传的文件
        
        Args:
            file_content: 文件内容
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        # 确保上传目录存在
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{file_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 写入文件
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path