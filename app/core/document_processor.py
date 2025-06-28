import os
from typing import Dict, List, Optional

class DocumentProcessor:
    """简单的文档处理器"""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        self.documents: Dict[str, str] = {}  # file_id -> content
        
    def process_file(self, file_id: str, file_path: str) -> bool:
        """处理文件，提取文本内容
        
        Args:
            file_id: 文件ID
            file_path: 文件路径
            
        Returns:
            bool: 是否处理成功
        """
        try:
            # 简单起见，先只处理文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.documents[file_id] = content
            return True
        except Exception as e:
            print(f"处理文件失败: {str(e)}")
            return False
            
    def search_documents(self, query: str) -> List[Dict[str, str]]:
        """搜索文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Dict[str, str]]: 匹配的文档片段列表
        """
        results = []
        for file_id, content in self.documents.items():
            # 简单的文本匹配
            if query.lower() in content.lower():
                # 找到匹配位置的上下文
                start = max(0, content.lower().find(query.lower()) - 100)
                end = min(len(content), start + 300)
                context = content[start:end]
                
                results.append({
                    "file_id": file_id,
                    "context": context,
                    "score": 1.0  # 简单起见，先使用固定分数
                })
        
        return results