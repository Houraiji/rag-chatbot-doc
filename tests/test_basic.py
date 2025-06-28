from fastapi.testclient import TestClient
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

# 创建测试客户端
client = TestClient(app)

def test_root():
    """测试根路由"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "欢迎使用RAG问答系统~"

def test_health():
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()
    assert response.json()["version"] == "0.1.0" 