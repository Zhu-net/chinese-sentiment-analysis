# 使用官方 Python 镜像（slim 版本更小）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件（利用 Docker 缓存）
COPY requirements.txt .

# 安装依赖（Docker 中用 CPU 版 torch，更通用）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务（自动检查并下载模型）
CMD ["python", "src/deploy/start.py"]
