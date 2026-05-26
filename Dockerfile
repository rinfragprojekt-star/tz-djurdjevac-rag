# Use stable Python version compatible with your dependencies
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Cloud Run expects the app to listen on port 8080
ENV PORT=8080

# Start the app using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
