import asyncio
import os
import sys
from dotenv import load_dotenv
from app.database.orchestrator import RAGPipelineOrchestrator

# 确保能导入 app 模块 (如果你的 main.py 在根目录下，通常不需要这行，为了保险加上)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量 (数据库连接等)
load_dotenv()
USERNAME = os.getenv("DB_USERNAME", "root")
PASSWORD = os.getenv("DB_PASSWORD", "")
HOSTNAME = os.getenv("DB_HOSTNAME", "localhost")
PORT = os.getenv("DB_PORT", "3306")
DATABASE = os.getenv("DB_DATABASE", "jory")
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)


if __name__ == '__main__':
    orchestrator = RAGPipelineOrchestrator(db_url=MYSQL_URI)
    file_path = r"F:\myRAG\parse\\test.pdf"
    asyncio.run(orchestrator.run_pipeline(file_path))
