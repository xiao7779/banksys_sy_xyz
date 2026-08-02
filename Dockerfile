FROM python:3.11-slim

WORKDIR /app

# 安装依赖（国内可指定镜像源加速）
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY app.py .

# 创建模型目录（预训练模型需由用户挂载或训练后生成）
RUN mkdir -p models

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8888", "--server.address=0.0.0.0"]
