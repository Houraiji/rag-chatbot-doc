import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.utils.memory import RedisMemoryManager, CustomRedisChatMessageHistory


@pytest.fixture
def memory_manager():
    """创建内存管理器实例"""
    manager = RedisMemoryManager()
    yield manager
    
    # 清理测试会话
    for session_id in manager.list_sessions():
        if session_id.startswith("test_"):
            manager.delete_session(session_id)


def test_session_management(memory_manager):
    """测试会话管理功能"""
    # 创建会话
    session_id = "test_" + memory_manager.create_session()
    
    # 检查会话是否存在
    assert memory_manager.session_exists(session_id)
    
    # 更新会话信息
    memory_manager.update_session_info(session_id, {"test_key": "test_value"})
    
    # 获取会话信息
    session_info = memory_manager.get_session_info(session_id)
    assert "test_key" in session_info
    assert session_info["test_key"] == "test_value"
    
    # 删除会话
    memory_manager.delete_session(session_id)
    assert not memory_manager.session_exists(session_id)


def test_chat_message_history():
    """测试聊天消息历史记录"""
    # 创建聊天历史记录
    session_id = "test_chat_history"
    chat_history = CustomRedisChatMessageHistory(session_id=session_id)
    
    # 添加消息
    chat_history.add_message(HumanMessage(content="你好"))
    chat_history.add_message(AIMessage(content="你好！有什么可以帮助你的吗？"))
    
    # 获取消息
    messages = chat_history.messages
    assert len(messages) == 2
    assert messages[0].content == "你好"
    assert messages[1].content == "你好！有什么可以帮助你的吗？"
    
    # 清空消息
    chat_history.clear()
    assert len(chat_history.messages) == 0