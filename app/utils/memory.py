import json
import time
import uuid
from typing import Dict, List, Optional, Any

import redis
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB


class RedisMemoryManager:
    """Redis对话记忆管理器"""
    
    def __init__(self):
        """初始化Redis连接"""
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True
        )
        
    def create_session(self) -> str:
        """创建新的会话ID"""
        session_id = str(uuid.uuid4())
        self.redis_client.hset(f"session:{session_id}", "created_at", time.time())
        return session_id
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return self.redis_client.exists(f"session:{session_id}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        if not self.session_exists(session_id):
            return {}
        return self.redis_client.hgetall(f"session:{session_id}")
    
    def update_session_info(self, session_id: str, info: Dict[str, Any]):
        """更新会话信息"""
        if self.session_exists(session_id):
            self.redis_client.hset(f"session:{session_id}", mapping=info)
    
    def delete_session(self, session_id: str):
        """删除会话"""
        if self.session_exists(session_id):
            # 删除会话信息
            self.redis_client.delete(f"session:{session_id}")
            # 删除会话历史消息
            self.redis_client.delete(f"chat_history:{session_id}")
    
    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        sessions = []
        for key in self.redis_client.scan_iter("session:*"):
            sessions.append(key.split(":", 1)[1])
        return sessions
    
    def clean_old_sessions(self, max_age_seconds: int = 86400):
        """清理旧会话（默认24小时）"""
        current_time = time.time()
        for session_id in self.list_sessions():
            session_info = self.get_session_info(session_id)
            created_at = float(session_info.get("created_at", 0))
            if current_time - created_at > max_age_seconds:
                self.delete_session(session_id)


class CustomRedisChatMessageHistory(BaseChatMessageHistory):
    """自定义Redis聊天消息历史记录"""
    
    def __init__(self, session_id: str, ttl: Optional[int] = None):
        """初始化Redis聊天消息历史记录
        
        Args:
            session_id: 会话ID
            ttl: 消息过期时间（秒）
        """
        self.session_id = session_id
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB
        )
        self.key = f"chat_history:{session_id}"
        self.ttl = ttl
    
    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        raw_messages = self.redis_client.lrange(self.key, 0, -1)
        messages = []
        for raw_message in raw_messages:
            message_dict = json.loads(raw_message)
            if message_dict["type"] == "human":
                messages.append(HumanMessage(content=message_dict["content"]))
            elif message_dict["type"] == "ai":
                messages.append(AIMessage(content=message_dict["content"]))
            elif message_dict["type"] == "system":
                messages.append(SystemMessage(content=message_dict["content"]))
        return messages
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        message_dict = {
            "type": message.type,
            "content": message.content
        }
        self.redis_client.rpush(self.key, json.dumps(message_dict))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)
    
    def clear(self) -> None:
        """清空消息"""
        self.redis_client.delete(self.key)


class CustomConversationBufferMemory(ConversationBufferMemory):
    """自定义对话缓冲记忆"""
    
    def __init__(
        self,
        session_id: str,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        max_token_limit: Optional[int] = None,
        ttl: Optional[int] = None,
        system_message: Optional[str] = None
    ):
        """初始化自定义对话缓冲记忆
        
        Args:
            session_id: 会话ID
            memory_key: 记忆键名
            return_messages: 是否返回消息对象
            max_token_limit: 最大token限制
            ttl: 消息过期时间（秒）
            system_message: 系统消息
        """
        chat_memory = CustomRedisChatMessageHistory(session_id=session_id, ttl=ttl)
        super().__init__(
            chat_memory=chat_memory,
            memory_key=memory_key,
            return_messages=return_messages,
        )
        self.max_token_limit = max_token_limit
        
        # 如果有系统消息，添加到历史记录
        if system_message:
            self.chat_memory.add_message(SystemMessage(content=system_message))
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存上下文"""
        super().save_context(inputs, outputs)
        
        # 如果设置了最大token限制，则进行截断
        if self.max_token_limit:
            self._truncate_history()
    
    def _truncate_history(self) -> None:
        """截断历史记录以满足token限制"""
        # 这里可以实现更复杂的截断逻辑，如摘要等
        # 目前简单实现为保留最近的消息
        messages = self.chat_memory.messages
        if not messages:
            return
        
        # 简单估算token数（实际应用中应使用tiktoken等库精确计算）
        total_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)
        
        # 如果超出限制，从最早的消息开始删除
        if total_tokens > self.max_token_limit:
            # 删除最早的消息，直到满足token限制
            self.chat_memory.clear()
            
            # 从最新的消息开始添加，直到接近但不超过限制
            current_tokens = 0
            for msg in reversed(messages):
                msg_tokens = len(msg.content.split()) * 1.3
                if current_tokens + msg_tokens > self.max_token_limit * 0.9:
                    break
                current_tokens += msg_tokens
                self.chat_memory.add_message(msg) 