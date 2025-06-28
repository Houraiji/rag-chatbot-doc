from setuptools import setup, find_packages

setup(
    name="rag-smartchat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "full": [
            "langchain>=0.0.267",
            "langchain-community>=0.0.10",
            "langchain-openai>=0.0.2",
            "openai>=1.0.0",
            "chromadb>=0.4.15",
            "redis>=4.6.0",
            "sentence-transformers>=2.2.2",
            "pypdf>=3.15.1",
            "tiktoken>=0.4.0",
            "pydantic>=2.4.2",
        ]
    }
) 