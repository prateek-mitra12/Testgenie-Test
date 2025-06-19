<<<<<<< HEAD
# Use official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
=======
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
>>>>>>> 7c9d2084d4bb5c558a74d70ce0054605a364f593

# Set work directory
WORKDIR /app

<<<<<<< HEAD
# Install system dependencies for Python packages (docx, pandas, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        python3-dev \
        default-libmysqlclient-dev \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        zlib1g-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install tesseract and poppler-utils for PDF/image support (optional, remove if not needed)
# RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit config: prevent browser auto-opening, set server address
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSEND=false

# Run Streamlit app
CMD ["streamlit", "run", "main.py"]
=======
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --use-pep517

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"] 
>>>>>>> 7c9d2084d4bb5c558a74d70ce0054605a364f593
