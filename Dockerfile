FROM python:3.12-slim

ARG PORT_ARG=3000
ENV PORT=${PORT_ARG} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.5.14

COPY . .

RUN uv pip install --system --no-cache -e . && \
    crawl4ai-setup

# Ensure the application can start
ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE ${PORT}

CMD ["python", "src/crawl4ai_mcp.py"]
