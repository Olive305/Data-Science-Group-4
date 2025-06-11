FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY analysis_code/ analysis_code/
COPY data/ /data/
COPY data_extraction/ data_extraction/
COPY models/ models/
COPY paper/ paper/
COPY testing/ testing/

ENV PYTHONPATH=/app

ENTRYPOINT ["/bin/bash"]
