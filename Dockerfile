FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install openenv from pip (it's required by constraints)
# openenv-core was included in requirements.txt

COPY . .

# Hugging face spaces map port 7860
EXPOSE 7860

# We use uvicorn to serve the environment
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
