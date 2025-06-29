import os
from typing import Dict, List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings  # 更新 OpenAIEmbeddings 导入
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # 更新 Document 导入
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chromadb.config import Settings

class DocumentRetriever:
    """文档处理器，支持文本和PDF文件"""
    
    def __init__(self, upload_dir: str, index_dir: str, openai_api_key: str, openai_api_base: str):
        self.upload_dir = upload_dir
        self.index_dir = index_dir
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 向量存储
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base
        )
        
        # 初始化或加载向量存储
        try:
            if os.path.exists(index_dir):
                self.vector_store = Chroma(
                    persist_directory=index_dir,
                    embedding_function=self.embeddings,
                )
            else:
                self.vector_store = Chroma.from_documents(
                    documents=[],  # 初始化空文档列表
                    embedding=self.embeddings,
                    persist_directory=index_dir,
                )
            
            # 创建检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
        except Exception as e:
            print(f"初始化向量存储失败: {str(e)}")
            raise

    def _load_document(self, file_path: str) -> List[Document]:
        """加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 文档列表
        """
        # 根据文件扩展名选择加载器
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in ['.txt', '.md', '.csv']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")
            
        return loader.load()
    
    async def retrieve(self, query: str) -> List[Document]:
        """检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 相关文档列表
        """
        try:
            return await self.retriever.ainvoke(query)
        except Exception as e:
            print(f"检索失败: {str(e)}")
            return []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """同步检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 相关文档列表
        """
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"检索失败: {str(e)}")
            return [] 
            

    
    # def process_file(self, file_id: str, file_path: str) -> bool:
    #     """处理文件，提取文本并创建向量索引"""
    #     try:
    #         # 根据文件类型选择不同的处理方法
    #         file_ext = os.path.splitext(file_path)[1].lower()
            
    #         if file_ext == '.pdf':
    #             text = self._process_pdf(file_path)
    #         else:  # .txt, .md, .csv
    #             text = self._process_text(file_path)
            
    #         # 分割文本
    #         chunks = self.text_splitter.split_text(text)
            
    #         # 创建文档对象
    #         documents = [
    #             Document(
    #                 page_content=chunk,
    #                 metadata={
    #                     "file_id": file_id,
    #                     "file_path": file_path,
    #                     "chunk_id": i
    #                 }
    #             ) for i, chunk in enumerate(chunks)
    #         ]
            
    #         # 添加到向量存储
    #         self.vector_store.add_documents(documents)
            
    #         return True
            
    #     except Exception as e:
    #         print(f"处理文件失败: {str(e)}")
    #         return False
    
    # def _process_pdf(self, file_path: str) -> str:
    #     """处理PDF文件"""
    #     reader = PdfReader(file_path)
    #     text = ""
    #     for page in reader.pages:
    #         text += page.extract_text() + "\n\n"
    #     return text
    
    # def _process_text(self, file_path: str) -> str:
    #     """处理文本文件"""
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         return f.read()
    
    # def search_documents(self, query: str, top_k: int = 2) -> List[Dict]:
    #     """搜索相关文档"""
    #     try:
    #         # 使用向量存储搜索
    #         results = self.vector_store.similarity_search_with_score(
    #             query=query,
    #             k=top_k
    #         )
            
    #         return [{
    #             "content": doc.page_content,
    #             "metadata": doc.metadata,
    #             "score": score
    #         } for doc, score in results]
            
    #     except Exception as e:
    #         print(f"搜索失败: {str(e)}")
    #         return []