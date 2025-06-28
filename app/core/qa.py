import os
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.core.config import (
    OPENAI_API_KEY, 
    OPENAI_API_BASE,
    OPENAI_MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CHROMA_PERSIST_DIRECTORY
)
from app.core.document_processor import DocumentProcessor
from app.core.retriever import HybridRetriever
from app.utils.memory import RedisMemoryManager, CustomConversationBufferMemory


# 定义提示模板
CONDENSE_QUESTION_PROMPT = """
根据以下对话历史和最新问题，生成一个独立的问题，该问题应包含所有相关上下文信息，以便能够在没有对话历史的情况下被理解和回答。

对话历史:
{chat_history}

最新问题: {question}

独立问题:
"""

QA_PROMPT = """
你是一个专业的知识助手，基于提供的文档内容回答用户问题。请遵循以下规则：

1. 只基于提供的文档内容回答问题，不要编造信息
2. 如果文档内容不足以回答问题，请明确说明无法回答或需要更多信息
3. 回答应该简洁明了，直接针对问题要点
4. 回答中应包含文档中的关键事实和数据
5. 使用用户的语言回答问题

问题: {question}

相关文档内容:
{context}

回答:
"""


class DocumentQA:
    """文档问答系统"""
    
    def __init__(self):
        """初始化文档问答系统"""
        self.document_processor = DocumentProcessor()
        self.retriever = HybridRetriever()
        self.memory_manager = RedisMemoryManager()
        self.qa_chain = None
        self.index_path = CHROMA_PERSIST_DIRECTORY
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=OPENAI_MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # 加载现有索引
        self.load_existing_index()
    
    def load_and_index_pdf(self, file_path: str) -> None:
        """加载和索引PDF文件
        
        Args:
            file_path: PDF文件路径
        """
        try:
            # 处理文档
            documents = self.document_processor.process_file(file_path)
            
            # 添加到向量数据库
            self.retriever.add_documents(documents)
            
            # 创建QA链
            self._create_qa_chain()
            
            print(f"成功处理文件: {file_path}, 共 {len(documents)} 个文档块")
            return True
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            raise e
    
    def load_existing_index(self) -> bool:
        """加载现有索引
        
        Returns:
            是否成功加载
        """
        try:
            if os.path.exists(self.index_path):
                # 加载向量数据库
                if self.retriever.load_existing_index():
                    # 创建QA链
                    self._create_qa_chain()
                    return True
            return False
        except Exception as e:
            print(f"加载索引时出错: {str(e)}")
            return False
    
    def _create_qa_chain(self) -> None:
        """创建问答链"""
        # 创建提示模板
        condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
        qa_prompt = PromptTemplate.from_template(QA_PROMPT)
        
        # 创建QA链
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.vector_db.as_retriever(),
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
        )
    
    def answer_question(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """回答问题
        
        Args:
            question: 用户问题
            session_id: 会话ID，用于多轮对话
            return_sources: 是否返回来源文档
            
        Returns:
            回答结果
        """
        start_time = time.time()
        
        # 确保QA链已创建
        if not self.qa_chain:
            if not self.load_existing_index():
                raise Exception("索引不存在，请先添加文档")
        
        # 处理会话ID
        if not session_id:
            session_id = self.memory_manager.create_session()
        elif not self.memory_manager.session_exists(session_id):
            self.memory_manager.create_session()
        
        # 创建对话记忆
        memory = CustomConversationBufferMemory(
            session_id=session_id,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,
            system_message="你是一个专业的知识助手，帮助用户回答关于文档的问题。"
        )
        
        # 使用自定义检索器
        self.qa_chain.retriever = self.retriever
        
        # 回答问题
        result = self.qa_chain({
            "question": question,
            "chat_history": memory.chat_memory.messages
        })
        
        # 保存对话历史
        memory.save_context({"question": question}, {"answer": result["answer"]})
        
        # 处理返回结果
        end_time = time.time()
        processing_time = end_time - start_time
        
        response = {
            "answer": result["answer"],
            "session_id": session_id,
            "processing_time": processing_time
        }
        
        # 如果需要返回来源文档
        if return_sources and "source_documents" in result:
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            response["sources"] = sources
        
        return response