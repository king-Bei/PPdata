import os
import base64
import binascii
import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # 新增
import io
import csv
import re
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["https://jollify.voyage.com.tw:8443"])  # 新增

# 設定 log 檔案
import logging

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")


def _dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        normalized = os.path.normpath(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def resolve_tessdata_dir():
    """Locate tessdata directory that contains ocrb.traineddata."""
    env_prefix = os.environ.get("TESSDATA_PREFIX")
    env_candidates = []
    if env_prefix:
        env_prefix = os.path.abspath(env_prefix)
        env_candidates.append(env_prefix)
        if os.path.basename(env_prefix) != "tessdata":
            env_candidates.append(os.path.join(env_prefix, "tessdata"))

    candidates = _dedupe_preserve_order([
        *env_candidates,
        DEFAULT_TESSDATA_DIR,
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
    ])

    for candidate in candidates:
        ocrb_file = os.path.join(candidate, "ocrb.traineddata")
        if os.path.exists(ocrb_file):
            logging.info(f"✅ 訓練數據文件已存在: {ocrb_file}")
            return candidate

    logging.error("❌ 找不到 ocrb.traineddata，請確認 tessdata 目錄配置是否正確")
    raise FileNotFoundError(
        "❌ 無法找到 ocrb.traineddata。請確認已將檔案放置於 apippno/tessdata 或設定正確的 TESSDATA_PREFIX。"
    )


tessdata_dir = resolve_tessdata_dir()
OCR_CONFIG = f'--tessdata-dir "{tessdata_dir}"'


@app.route('/')
def home():
    logging.info("📢 API 首頁被訪問")
    return render_template('index.html')


@app.route('/api/ocr', methods=['POST'])
def ocr_in_memory():
    """OCR API: 接收圖片檔案或 Base64 字串並提取護照 MRZ 資料"""

    logging.info("📥 收到 OCR API 請求")

    file = request.files.get('image')
    image_data = None
    source_desc = None

    if file and file.filename != '':
        image_data = file.read()
        source_desc = f"上傳檔案 {file.filename}"
        logging.info(f"📸 影像 {file.filename} 已讀取，大小: {len(image_data)} bytes")
    else:
        # 嘗試從 JSON 或表單欄位讀取 Base64 影像
        payload = request.get_json(silent=True) or {}
        base64_string = (
            payload.get('image_base64')
            or payload.get('image')
            or request.form.get('image_base64')
            or request.form.get('image')
        )

        if base64_string:
            try:
                # 支援 data URL 格式，如 data:image/png;base64,XXXX
                if ',' in base64_string:
                    base64_string = base64_string.split(',', 1)[1]

                image_data = base64.b64decode(base64_string, validate=True)
                source_desc = "Base64 字串"
                logging.info(f"🧾 已解碼 Base64 影像，大小: {len(image_data)} bytes")
            except (binascii.Error, ValueError) as decode_error:
                logging.warning(f"⚠️ Base64 影像解析失敗: {decode_error}")
                return jsonify({'status': 'error', 'message': 'Invalid base64 image data'}), 400

    if image_data is None:
        logging.warning("⚠️ 沒有上傳圖片或 Base64 資料")
        return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

    try:
        # 以 Pillow 開啟影像
        image = Image.open(io.BytesIO(image_data))
        logging.info(f"🖼️ 影像來源: {source_desc} 已成功載入")

        # 影像預處理
        processed_image = preprocess_image_pil(image)
        logging.info("🔄 影像已預處理完成")

        # OCR 辨識
        detected_text = pytesseract.image_to_string(processed_image, lang='ocrb', config=OCR_CONFIG)
        logging.info(f"✅ OCR 辨識完成，識別結果: {detected_text[:50]}...")

        # 解析 MRZ 資料
        mrz_data = extract_mrz_data(detected_text)

        return jsonify({'status': 'success', 'data': mrz_data}), 200

    except ValueError as ve:
        logging.error(f"❌ MRZ 解析錯誤: {str(ve)}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400

    except Exception as e:
        logging.error(f"❌ OCR 過程發生錯誤: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Internal Server Error'}), 500


def detect_mrz_region(image):
    """
    使用 OpenCV 自動偵測 MRZ 區域
    :param image: PIL Image
    :return: PIL Image (MRZ 區域)
    """
    # 轉成 OpenCV 格式
    cv_img = np.array(image)
    if cv_img.ndim == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    # 增強對比
    cv_img = cv2.equalizeHist(cv_img)
    # 二值化
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 邊緣偵測
    edges = cv2.Canny(thresh, 100, 200)
    # 膨脹讓線條更明顯
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    # 找輪廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = cv_img.shape
    mrz_candidate = None
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch
        # MRZ 通常很寬很扁，且在下方
        if aspect_ratio > 5 and ch > h * 0.05 and y > h * 0.5:
            if mrz_candidate is None or ch * cw > mrz_candidate[2] * mrz_candidate[3]:
                mrz_candidate = (x, y, cw, ch)
    if mrz_candidate:
        x, y, cw, ch = mrz_candidate
        mrz_img = cv_img[y:y+ch, x:x+cw]
        return Image.fromarray(mrz_img)
    else:
        # 找不到就 fallback 用下方 25%
        width, height = image.size
        return image.crop((0, int(height * 0.75), width, height))


def preprocess_image_pil(image):
    """
    使用 PIL 和 OpenCV 處理影像，提高 OCR 的辨識精度
    """
    gray = ImageOps.grayscale(image)
    mrz_region = detect_mrz_region(gray)
    return mrz_region


def extract_mrz_data(text):
    """
    提取 MRZ 資料並解析
    """
    lines = text.strip().split('\n')

    # 確保至少有兩行 MRZ
    if len(lines) < 2:
        raise ValueError("MRZ 資料不足，無法解析")

    mrz_line1 = lines[-2].replace(' ', '')
    mrz_line2 = lines[-1].replace(' ', '')

    return parse_mrz(mrz_line1, mrz_line2)

def parse_mrz(line1, line2):
    """
    解析 MRZ 資料
    """
    try:
        names = line1[5:].split('<<')
        last_name = names[0].replace('<', ' ').strip()
        first_name = names[1].replace('<', ' ').strip()

        passport_number = line2[0:9].strip()
        nationality = line2[10:13].strip()
        birth_date = format_date(line2[13:19])
        gender = 'M' if line2[20] == 'M' else 'F'
        expiry_date = format_date(line2[21:27])
        ID_number = line2[28:38].strip()

        return {
            "last_name": last_name,
            "first_name": first_name,
            "passport_number": passport_number,
            "nationality": nationality,
            "birth_date": birth_date,
            "gender": gender,
            "expiry_date": expiry_date,
            "ID_number": ID_number
        }
    except Exception as e:
        raise ValueError(f"MRZ 解析失敗: {str(e)}")



def format_date(date_string):
    """
    將 MRZ 日期格式 (YYMMDD) 轉換為 YYYY/MM/DD
    """
    if not re.match(r'^\d{6}$', date_string):
        return "Invalid Date"

    year = int(date_string[:2])
    month = int(date_string[2:4])
    day = int(date_string[4:6])

    # 判斷 1900 或 2000 年代
    full_year = 2000 + year if year < 50 else 1900 + year

    try:
        return datetime(full_year, month, day).strftime("%Y/%m/%d")
    except ValueError:
        return "Invalid Date"




@app.route('/api/ocr-csv', methods=['POST'])
def ocr_in_memory_csv():
    """
    接收使用者上傳的圖片，在記憶體中進行 OCR 後，將結果寫入 CSV。
    """
    logging.info("📥 收到 OCR CSV API 請求")

    if 'image' not in request.files:
        logging.warning("⚠️ 沒有上傳圖片")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        image_data = file.read()
        logging.info(f"📸 影像 {file.filename} 已讀取，大小: {len(image_data)} bytes")

        image = Image.open(io.BytesIO(image_data))
        logging.info("🖼️ 影像已成功載入")

        # 預處理 + OCR
        processed_image = preprocess_image_pil(image)
        detected_text = pytesseract.image_to_string(processed_image, lang='ocrb', config=OCR_CONFIG)
        logging.info(f"✅ OCR 辨識完成，識別結果: {detected_text[:50]}...")

        # 準備寫入 CSV
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        output_dir = '/tmp/tmp_output'
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"output_{date_str}.csv")

        append_header = not os.path.isfile(output_csv)
        row = {
            'date': date_str,
            'time': now.strftime("%H:%M:%S"),
            'filename': file.filename,
            'detected_text': detected_text
        }

        with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['date', 'time', 'filename', 'detected_text'])
            if append_header:
                writer.writeheader()
            writer.writerow(row)

        logging.info(f"📄 OCR 結果已寫入 CSV: {output_csv}")

        return jsonify({
            'message': f'處理完成，寫入至 {output_csv}',
            'detected_text': detected_text
        }), 200

    except Exception as e:
        logging.error(f"❌ OCR CSV 處理過程發生錯誤: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    logging.info(f"🚀 啟動 Flask 伺服器，監聽 PORT {port}")
    app.run(host='0.0.0.0', port=port)
