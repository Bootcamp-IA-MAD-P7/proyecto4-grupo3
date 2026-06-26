FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY . .

RUN uv sync --no-cache

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]