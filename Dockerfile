# Use slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System-level dependencies (for torch + spaCy)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download the large spaCy model
RUN python -m spacy download en_core_web_lg

# Optional: If you're downloading IndicNER model in runtime, no need to do anything.
# If not, and you're using HuggingFace model manually, you can pre-cache it (optional).
# e.g., RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('ai4bharat/IndicNER')"

# Copy project files
COPY . /app

# Expose app port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
