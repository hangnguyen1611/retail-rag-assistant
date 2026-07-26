FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend (Streamlit) dùng chung image này, override command trong
# docker-compose.yml (xem service "frontend") thay vì build image riêng.