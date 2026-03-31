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
import json
import secrets
from uuid import uuid4
from datetime import datetime, timezone

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # For session support
CORS(app, origins=["https://jollify.voyage.com.tw:8443"])  # 新增

# 設定 log 檔案
import logging

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")
APP_KEYS_FILE = os.path.join(BASE_DIR, "app_keys.json")

# 模擬使用者資料（實際環境應使用資料庫與 Hash 密碼）
USERS = {
    "admin": "admin123"
}


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


def load_app_keys():
    if not os.path.exists(APP_KEYS_FILE):
        return []
    try:
        with open(APP_KEYS_FILE, 'r', encoding='utf-8') as file:
            data = json.load(file) or []
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        logging.warning("⚠️ app_keys.json 內容格式錯誤，將重新初始化")
    except Exception as exc:
        logging.error(f"❌ 無法讀取 app_keys.json: {exc}")
    return []


def save_app_keys(data):
    try:
        with open(APP_KEYS_FILE, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.error(f"❌ 無法寫入 app_keys.json: {exc}")
        raise


def find_app_key(apps, app_id):
    for app_key in apps:
        if app_key.get('id') == app_id:
            return app_key
    return None


def normalize_period_fields(payload):
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    if start_date:
        start_date = start_date.strip()
    if end_date:
        end_date = end_date.strip()
    return start_date or None, end_date or None


def validate_key_payload(payload):
    key_type = payload.get('key_type')
    if key_type not in {'permanent', 'periodic'}:
        raise ValueError('key_type 必須為 permanent 或 periodic')

    start_date, end_date = normalize_period_fields(payload)
    if key_type == 'periodic':
        if not start_date or not end_date:
            raise ValueError('週期金鑰需填寫開始與結束日期')
        if start_date > end_date:
            raise ValueError('結束日期需晚於開始日期')
    return start_date, end_date


@app.route('/')
def home():
    from flask import session, redirect, url_for
    if 'user' not in session:
        logging.info("🚶 未登入，重導向至登入頁面")
        return redirect(url_for('login_page'))
    logging.info("📢 API 首頁被訪問")
    return render_template('index.html')


@app.route('/login')
def login_page():
    logging.info("🔐 登入頁面被訪問")
    return render_template('login.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if USERS.get(username) == password:
        from flask import session
        session['user'] = username
        logging.info(f"✅ 使用者 {username} 登入成功")
        return jsonify({'status': 'success', 'message': '登入成功'})
    
    logging.warning(f"❌ 使用者 {username} 登入失敗")
    return jsonify({'status': 'error', 'message': '帳號或密碼錯誤'}), 401


@app.route('/api/logout', methods=['POST'])
def api_logout():
    from flask import session
    session.pop('user', None)
    return jsonify({'status': 'success', 'message': '已登出'})


@app.route('/api/app-keys', methods=['GET'])
def list_app_keys():
    logging.info("📋 讀取金鑰設定清單")
    return jsonify({'status': 'success', 'data': load_app_keys()})


@app.route('/api/app-keys', methods=['POST'])
def create_app_key():
    payload = request.get_json(force=True)
    logging.info("➕ 建立新的來源金鑰")
    for field in ('source_name', 'app_name'):
        if not payload.get(field):
            return jsonify({'status': 'error', 'message': f'{field} 為必填欄位'}), 400

    try:
        start_date, end_date = validate_key_payload(payload)
    except ValueError as err:
        return jsonify({'status': 'error', 'message': str(err)}), 400

    status = payload.get('status', 'active')
    if status not in {'active', 'disabled'}:
        return jsonify({'status': 'error', 'message': 'status 僅能為 active 或 disabled'}), 400

    apps = load_app_keys()
    now_utc = datetime.now(timezone.utc).isoformat()
    new_app = {
        'id': str(uuid4()),
        'source_name': payload.get('source_name').strip(),
        'app_name': payload.get('app_name').strip(),
        'api_key': payload.get('api_key') or secrets.token_urlsafe(24),
        'key_type': payload.get('key_type'),
        'start_date': start_date,
        'end_date': end_date,
        'status': status,
        'created_at': now_utc,
        'updated_at': now_utc,
    }
    apps.append(new_app)
    save_app_keys(apps)
    return jsonify({'status': 'success', 'data': new_app}), 201


@app.route('/api/app-keys/<app_id>', methods=['PUT'])
def update_app_key(app_id):
    payload = request.get_json(force=True)
    apps = load_app_keys()
    target = find_app_key(apps, app_id)
    if not target:
        return jsonify({'status': 'error', 'message': '找不到指定的來源金鑰'}), 404

    if 'key_type' in payload:
        try:
            start_date, end_date = validate_key_payload({**target, **payload})
        except ValueError as err:
            return jsonify({'status': 'error', 'message': str(err)}), 400
        target['key_type'] = payload['key_type']
        target['start_date'] = start_date
        target['end_date'] = end_date
    elif target.get('key_type') == 'periodic' and ('start_date' in payload or 'end_date' in payload):
        start_date = payload.get('start_date', target.get('start_date')) or ''
        end_date = payload.get('end_date', target.get('end_date')) or ''
        start_date = start_date.strip()
        end_date = end_date.strip()
        if not start_date or not end_date:
            return jsonify({'status': 'error', 'message': '週期金鑰需同時提供起訖日期'}), 400
        if start_date > end_date:
            return jsonify({'status': 'error', 'message': '結束日期需晚於開始日期'}), 400
        target['start_date'] = start_date
        target['end_date'] = end_date

    if 'status' in payload:
        if payload['status'] not in {'active', 'disabled'}:
            return jsonify({'status': 'error', 'message': 'status 僅能為 active 或 disabled'}), 400
        target['status'] = payload['status']

    for field in ('source_name', 'app_name', 'api_key'):
        if field in payload and payload[field]:
            target[field] = payload[field].strip() if isinstance(payload[field], str) else payload[field]

    target['updated_at'] = datetime.now(timezone.utc).isoformat()
    save_app_keys(apps)
    return jsonify({'status': 'success', 'data': target})


@app.route('/api/app-keys/generate', methods=['GET'])
def generate_api_key():
    return jsonify({'status': 'success', 'data': {'api_key': secrets.token_urlsafe(24)}})


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
    使用 OpenCV 形態學運算優化 MRZ 區域偵測
    """
    cv_img = np.array(image)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    # 1. 使用 Blackhat 形態學運算突出黑色文字
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. 計算梯度 (Sobel) 找出垂直邊緣
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # 3. 閉合運算填補字元間隙
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 4. 再次進行閉合與腐蝕，減少雜訊
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # 5. 尋找輪廓
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = gray.shape
    candidates = []
    for c in cnts:
        (x, y, cw, ch) = cv2.boundingRect(c)
        ar = cw / float(ch)
        crWidth = cw / float(w)

        # MRZ 通常寬度佔比 > 70%，高寬比 > 5
        if ar > 5 and crWidth > 0.6:
            # 偏向影像下方 (護照 MRZ 位置)
            pX = x + (cw / 2)
            pY = y + (ch / 2)
            if pY > h * 0.5:
                candidates.append((x, y, cw, ch))

    if candidates:
        # 選擇最下方的候選區域
        candidates.sort(key=lambda b: b[1], reverse=True)
        (x, y, cw, ch) = candidates[0]
        
        # 稍微放大邊界
        p = 10
        rx = max(0, x - p)
        ry = max(0, y - p)
        rw = min(w, cw + 2 * p)
        rh = min(h, ch + 2 * p)
        
        return Image.fromarray(gray[ry:ry + rh, rx:rx + rw])
    else:
        # Fallback
        logging.warning("⚠️ 無法精確偵測 MRZ 區域，使用預設裁剪")
        return Image.fromarray(gray[int(h * 0.7):, :])


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

def mrz_checksum(data):
    """計算 MRZ 檢查碼 (ICAO 9303 標準)"""
    weight = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if '0' <= char <= '9':
            val = ord(char) - ord('0')
        elif 'A' <= char <= 'Z':
            val = ord(char) - ord('A') + 10
        elif char == '<':
            val = 0
        else:
            val = 0
        total += val * weight[i % 3]
    return total % 10

def parse_mrz(line1, line2):
    """
    解析 MRZ 資料並進行 Checksum 驗證
    """
    try:
        # 移除非法字元
        line1 = re.sub(r'[^A-Z0-9<]', '', line1.upper())
        line2 = re.sub(r'[^A-Z0-9<]', '', line2.upper())

        # 基本格式檢查
        if len(line1) < 44 or len(line2) < 44:
            raise ValueError("MRZ 長度不足")

        names = line1[5:44].split('<<')
        last_name = names[0].replace('<', ' ').strip()
        first_name = names[1].replace('<', ' ').strip() if len(names) > 1 else ""

        passport_number = line2[0:9].replace('<', '').strip()
        pass_check = line2[9] # Checksum for passport number

        birth_date_raw = line2[13:19]
        birth_check = line2[19]

        expiry_date_raw = line2[21:27]
        expiry_check = line2[27]

        # 驗證 Checksum
        valid_passport = str(mrz_checksum(line2[0:9])) == pass_check
        valid_birth = str(mrz_checksum(birth_date_raw)) == birth_check
        valid_expiry = str(mrz_checksum(expiry_date_raw)) == expiry_check

        nationality = line2[10:13].replace('<', '').strip()
        gender = 'M' if line2[20] == 'M' else 'F'
        ID_number = line2[28:42].replace('<', '').strip()

        return {
            "last_name": last_name,
            "first_name": first_name,
            "passport_number": passport_number,
            "nationality": nationality,
            "birth_date": format_date(birth_date_raw),
            "gender": gender,
            "expiry_date": format_date(expiry_date_raw),
            "ID_number": ID_number,
            "verification": {
                "passport": valid_passport,
                "birth_date": valid_birth,
                "expiry_date": valid_expiry,
                "overall": valid_passport and valid_birth and valid_expiry
            }
        }
    except Exception as e:
        logging.error(f"MRZ 解析失敗: {e}")
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
