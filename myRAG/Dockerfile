FROM python:3.11-slim

WORKDIR /app

# 关键优化：设置 PYTHONUNBUFFERED 以便及时看到日志
# 设置 PYTHONDONTWRITEBYTECODE 减少磁盘写入
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 强制单线程处理某些库，减少 CPU 争抢
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# 安装最基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    git \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # 清理缓存减小镜像体积 (对于 50G 硬盘很重要)
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN mkdir -p data/chroma_db data/images data/cache

EXPOSE 8000

# 针对低配机器：不使用 Gunicorn 的多 worker 模式，直接用 Uvicorn 单进程
# 任何多 worker 尝试都会导致 1核 CPU 爆炸和内存溢出
CMD ["python", "-m", "app.server"]
