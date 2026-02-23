import os
import sys
import time
import jwt
import requests
from datetime import datetime, timedelta, timezone

# 尝试加载 .env
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    print(f"Loading env from: {env_path}")
    load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed, using system env vars")

# 配置
RAG_HOST = os.getenv("RAG_HOST", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET")

def print_status(msg, status="INFO"):
    # Windows PowerShell 有时不支持 ANSI 颜色，简单打印即可
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅ ",
        "ERROR": "❌ ",
    }
    print(f"{prefix.get(status, '')}[{status}] {msg}")

def generate_test_token():
    if not JWT_SECRET:
        print_status("JWT_SECRET environment variable is missing!", "ERROR")
        return None
    
    payload = {
        "sub": "9999", # Test User ID
        "role": "admin",
        "username": "smoke_test_bot",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def check_health():
    url = f"{RAG_HOST}/health"
    print_status(f"Checking {url}...", "INFO")
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            json_data = res.json()
            if json_data.get('code') == 0:
                print_status("Health Check Passed", "SUCCESS")
                return True
            else:
                print_status(f"Health Check Logic Failed: {json_data}", "ERROR")
                return False
        else:
            print_status(f"Health Check Failed: Status {res.status_code}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Health Check Connection Failed: {e}", "ERROR")
        return False

def check_auth_upload():
    print_status("Checking Auth & Upload Endpoint...", "INFO")
    token = generate_test_token()
    if not token:
        return False
    
    url = f"{RAG_HOST}/api/v1/upload"
    headers = {"Authorization": f"Bearer {token}"}
    
    # 创建一个伪造的 PDF 内容
    files = {'file': ('test.pdf', b'%PDF-1.4 \n fake pdf content', 'application/pdf')}
    
    try:
        # 这里只是验证鉴权逻辑，不指望真的解析成功（因为是假PDF）
        # 只要不是 401 Unauthorized 就算通过鉴权测试
        res = requests.post(url, headers=headers, files=files, timeout=10)
        
        if res.status_code == 401:
            print_status("Auth Check Failed: 401 Unauthorized (Secret Mismatch or Middleware Error)", "ERROR")
            return False
        elif res.status_code == 200:
             print_status("Auth & Upload Check Passed", "SUCCESS")
             return True
        else:
            # 即使上传失败（比如500），只要不是401，说明鉴权通过了
            print_status(f"Auth Passed, but Upload Failed (Expected for fake PDF): {res.status_code}", "SUCCESS")
            return True
            
    except Exception as e:
        print_status(f"Auth Check Connection Failed: {e}", "ERROR")
        return False

if __name__ == "__main__":
    print_status("🚀 Starting Smoke Test...", "INFO")
    
    if not JWT_SECRET:
        print_status("⚠️  Warning: JWT_SECRET not found. Auth tests will be SKIPPED.", "INFO")
        print_status("Did you forget to create .env file in myRAG directory?", "INFO")
    
    health = check_health()
    auth = True
    
    if JWT_SECRET:
        auth = check_auth_upload()
    
    if health and auth:
        print_status("\n✨ All Systems Operational!", "SUCCESS")
        sys.exit(0)
    else:
        print_status("\n💀 System Verification Failed", "ERROR")
        sys.exit(1)
