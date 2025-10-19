# ---- Base image ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# OS build tools (some Python wheels need a compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data (keeps first run fast)
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','punkt','wordnet','omw-1.4','vader_lexicon']]"

# Copy app code
COPY . .

EXPOSE 8501
CMD ["streamlit","run","sentimentanalyzer.py","--server.address=0.0.0.0","--server.port=8501"]
