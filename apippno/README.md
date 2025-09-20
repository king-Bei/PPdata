# MRZ OCR API

一個使用 Python Flask、Tesseract OCR 及 OpenCV 的護照 MRZ（機器可讀區）影像辨識 API。

## 特色

- 提供 RESTful API 介面進行 MRZ 資料擷取與 OCR
- 自動偵測 MRZ 區域並進行影像預處理
- OCR 結果可匯出為 CSV
- 內建日誌紀錄方便除錯與監控

## 系統需求

- Python 3.7 以上
- Tesseract OCR（需包含 `ocrb` 訓練資料）
- OpenCV
- Pillow
- Flask
- pytesseract
- numpy

## 安裝方式

1. 下載專案原始碼：
    ```bash
    git clone https://github.com/yourusername/mrz-api.git
    cd mrz-api
    ```

2. 安裝相依套件：
    ```bash
    pip install -r requirements.txt
    ```

3. 請確認已安裝 Tesseract OCR，且 `ocrb.traineddata` 已放置於 tessdata 目錄下。

## 使用說明

啟動 Flask 伺服器：
```bash
python app.py
```

### API 端點

- `GET /`  
  首頁

- `POST /api/ocr`
  上傳圖片檔（表單欄位：`image`）或傳送 Base64 字串（JSON 欄位：`image_base64` 或 `image`），回傳解析後的 MRZ 資料（JSON 格式）

- `POST /api/ocr-csv`  
  上傳圖片檔（表單欄位：`image`），OCR 結果寫入 CSV 並回傳辨識文字

## 範例請求

```bash
curl -F "image=@passport.jpg" http://localhost:5002/api/ocr

# 或傳送 Base64 字串
curl -X POST http://localhost:5002/api/ocr \
  -H 'Content-Type: application/json' \
  -d '{"image_base64": "<BASE64_DATA>"}'
```

## 設定說明

- 若 tessdata 目錄非預設位置，請設定 `TESSDATA_PREFIX` 環境變數
- 預設 CSV 輸出目錄：`/tmp/tmp_output`

## 授權

MIT License
