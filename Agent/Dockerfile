FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制根目录下的依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目代码到容器（Docker Compose 会控制构建上下文）
COPY . .

# 关键：设置 Python 路径，确保 Agent 的代码能 import 其他模块
ENV PYTHONPATH=/app

# 进入 Agent 目录
WORKDIR /app/Agent

# 运行 main.py（这里假设它是直接启动一个 Web 服务器）
# 如果用的是 FastAPI，CMD 则是 ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# 如果用的是 Streamlit，则是 ["streamlit", "run", "main.py", "--server.port=8000", "--server.address=0.0.0.0"]
CMD ["python", "main.py"]