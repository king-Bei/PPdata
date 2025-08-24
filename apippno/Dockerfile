FROM python:3.9-slim

# 安裝基本套件 + Tesseract（不要裝 ocrb）
RUN apt-get update && \
    apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean

WORKDIR /app

# 複製應用程式與模板
COPY app.py requirements.txt templates/ /app/

# 確保 tessdata 目錄存在
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata

# 複製所有 tessdata 檔案（方便擴充）
COPY tessdata/* /usr/share/tesseract-ocr/4.00/tessdata/

RUN pip install --no-cache-dir -r requirements.txt

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV LANG=C.UTF-8

CMD ["python", "app.py"]
