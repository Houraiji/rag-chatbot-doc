from typing import List, Dict, Any, Optional, Tuple
import os
import re
import time
from collections import Counter

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from app.core.config import (
    OPENAI_API_KEY, 
    OPENAI_API_BASE, 
    OPENAI_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY, 
    EMBEDDING_MODEL_NAME,
    VECTOR_SEARCH_TOP_K,
    KEYWORD_SEARCH_TOP_K,
    HYBRID_ALPHA
)


class HybridRetriever:
    """混合检索器，结合向量检索和关键词检索"""
    
    def __init__(self):
        """初始化混合检索器"""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model=EMBEDDING_MODEL_NAME
        )
        
        # 向量数据库
        self.vector_db = None
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            try:
                self.vector_db = Chroma(
                    persist_directory=CHROMA_PERSIST_DIRECTORY,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"加载向量数据库时出错: {str(e)}")
        
        # 关键词检索器
        self.keyword_retriever = None
        
        # LLM用于查询扩展
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=OPENAI_MODEL_NAME,
            temperature=0
        )
        
        # 查询扩展检索器
        self.multi_query_retriever = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量数据库
        
        Args:
            documents: 文档列表
        """
        # 如果向量数据库不存在，则创建
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
        
        # 添加文档
        self.vector_db.add_documents(documents)
        
        # 更新关键词检索器
        self._update_keyword_retriever()
    
    def _update_keyword_retriever(self) -> None:
        """更新关键词检索器"""
        if self.vector_db is not None:
            # 获取所有文档
            all_docs = self.vector_db.get()
            if all_docs and "documents" in all_docs and all_docs["documents"]:
                documents = [
                    Document(page_content=text, metadata=metadata)
                    for text, metadata in zip(all_docs["documents"], all_docs["metadatas"])
                ]
                # 创建BM25检索器
                self.keyword_retriever = BM25Retriever.from_documents(documents)
                
                # 创建查询扩展检索器
                if self.vector_db is not None:
                    self.multi_query_retriever = MultiQueryRetriever.from_llm(
                        retriever=self.vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
                        llm=self.llm
                    )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """混合检索
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
            
        Returns:
            检索到的文档列表
        """
        start_time = time.time()
        
        # 确保向量数据库和关键词检索器已初始化
        if self.vector_db is None or self.keyword_retriever is None:
            if os.path.exists(CHROMA_PERSIST_DIRECTORY):
                try:
                    self.vector_db = Chroma(
                        persist_directory=CHROMA_PERSIST_DIRECTORY,
                        embedding_function=self.embeddings
                    )
                    self._update_keyword_retriever()
                except Exception as e:
                    raise Exception(f"加载向量数据库时出错: {str(e)}")
            else:
                raise Exception("向量数据库不存在，请先添加文档")
        
        # 向量检索
        vector_docs = self.vector_db.similarity_search_with_score(
            query, k=VECTOR_SEARCH_TOP_K
        )
        
        # 关键词检索
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)[:KEYWORD_SEARCH_TOP_K]
        
        # 查询扩展检索
        multi_query_docs = []
        if self.multi_query_retriever:
            try:
                multi_query_docs = self.multi_query_retriever.get_relevant_documents(query)
            except Exception as e:
                print(f"查询扩展检索时出错: {str(e)}")
        
        # 合并结果
        results = self._merge_results(vector_docs, keyword_docs, multi_query_docs, top_k)
        
        end_time = time.time()
        print(f"检索耗时: {end_time - start_time:.2f}秒")
        
        return results
    
    def _merge_results(
        self, 
        vector_docs: List[Tuple[Document, float]], 
        keyword_docs: List[Document],
        multi_query_docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """合并检索结果
        
        Args:
            vector_docs: 向量检索结果，包含相似度分数
            keyword_docs: 关键词检索结果
            multi_query_docs: 查询扩展检索结果
            top_k: 返回的文档数量
            
        Returns:
            合并后的文档列表
        """
        # 创建文档ID到文档的映射
        doc_map = {}
        scores = {}
        
        # 处理向量检索结果
        for doc, score in vector_docs:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            # 将相似度分数转换为0-1范围（越大越好）
            normalized_score = 1 - score  # 假设score是距离，越小越好
            scores[doc_id] = scores.get(doc_id, 0) + normalized_score * HYBRID_ALPHA
        
        # 处理关键词检索结果
        for i, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            # 根据排名计算分数
            keyword_score = (len(keyword_docs) - i) / len(keyword_docs) if keyword_docs else 0
            scores[doc_id] = scores.get(doc_id, 0) + keyword_score * (1 - HYBRID_ALPHA)
        
        # 处理查询扩展检索结果
        for doc in multi_query_docs:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            # 给予额外的加分
            scores[doc_id] = scores.get(doc_id, 0) + 0.1
        
        # 根据分数排序
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 获取前top_k个文档
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = doc_map[doc_id]
            # 添加分数到元数据
            doc.metadata["score"] = scores[doc_id]
            results.append(doc)
        
        return results
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档的唯一标识符"""
        # 使用文档内容的哈希值和文件ID（如果有）作为唯一标识符
        content_hash = hash(doc.page_content)
        file_id = doc.metadata.get("file_id", "")
        return f"{file_id}_{content_hash}"
    
    def load_existing_index(self) -> bool:
        """加载现有索引
        
        Returns:
            是否成功加载
        """
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            try:
                self.vector_db = Chroma(
                    persist_directory=CHROMA_PERSIST_DIRECTORY,
                    embedding_function=self.embeddings
                )
                self._update_keyword_retriever()
                return True
            except Exception as e:
                print(f"加载向量数据库时出错: {str(e)}")
                return False
        return False 