from typing import Dict, List, Any, Optional
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory

class QAChain:
    """基于 LangChain 的问答链"""
    def __init__(self, openai_api_key: str, openai_api_base: str, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.model_name = model_name

        # 初始化消息历史存储
        self.message_histories = {}

        # 创建llm
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            temperature=0.9
        )

        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的助手，请基于提供的上下文回答用户的问题。如果问题无法从上下文中得到答案，请说明。"),
            ("human", "我需要你回答关于以下内容的问题:\n\n上下文: {context}\n\n问题: {question}")
        ])

        # 格式上下文
        def _format_docs(docs: List[Document]) -> str:
            return "\n\n".join([doc.page_content for doc in docs])
        
        # QA链
        self.rag_chain = (
            {
                "context": lambda x: _format_docs(x["documents"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 创建带消息历史的可运行对象
        self.qa_with_history = RunnableWithMessageHistory(
            self.rag_chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.message_histories:
            self.message_histories[session_id] = InMemoryChatMessageHistory()
        return self.message_histories[session_id]
    
    async def answer_question(self, question: str, documents: List[Document], session_id: str) -> str:
        """回答问题
        
        Args:
            question: 用户问题
            documents: 相关文档列表
            session_id: 会话ID
            
        Returns:
            str: 回答内容
        """
        try:
            # 调用带历史的链
            response = await self.qa_with_history.ainvoke(
                {"question": question, "documents": documents},
                config={"configurable": {"session_id": session_id}}
            )
            
            return response
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return f"抱歉，生成回答时出错: {str(e)}"
    
    def get_message_history(self, session_id: str) -> List[Dict]:
        """获取会话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 消息历史列表
        """
        if session_id not in self.message_histories:
            return []
        
        messages = self.message_histories[session_id].messages
        return [{"role": msg.type, "content": msg.content} for msg in messages]
    
    def clear_history(self, session_id: str) -> bool:
        """清除会话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功清除
        """
        if session_id in self.message_histories:
            self.message_histories[session_id].clear()
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功删除
        """
        if session_id in self.message_histories:
            del self.message_histories[session_id]
            return True
        return False
        





