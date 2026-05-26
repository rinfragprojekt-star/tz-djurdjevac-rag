FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

# Dodaj timeout za Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]
