# Use official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for basic ML operations
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY cloud_backend/requirements.txt .

# Install Python dependencies (PyTorch, FastAPI, Uvicorn)
# Using the CPU-only PyTorch wheel to keep image size small for initial AWS testing
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the trained 2D static model weights (142MB)
COPY cloud_backend/weights/hagrid_resnet50_finetuned.pth ./weights/hagrid_resnet50_finetuned.pth

# Copy the backend application code
COPY cloud_backend/app/ ./app/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Include AWS Lambda Web Adapter to run standard FastAPI in AWS Lambda container
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000
ENV AWS_LWA_READINESS_CHECK_PATH=/health

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]