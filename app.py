import os
import base64
import binascii
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # 新增
import io
import csv
import re
import json
import secrets
import time
from uuid import uuid4
from datetime import datetime, timezone
from mrz import extract_mrz_data, parse_mrz, mrz_checksum, format_date
from mrz_image import split_mrz_lines, perspective_mrz_candidates

# Tesseract's OpenMP oversubscription can turn a short MRZ scan into a timeout.
os.environ.setdefault('OMP_THREAD_LIMIT', '1')

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # For session support
CORS(app, origins=["https://jollify.voyage.com.tw:8443"])  # 新增

# 設定 log 檔案
import logging

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")
APP_KEYS_FILE = os.path.join(BASE_DIR, "app_keys.json")
OCR_LANGUAGE = os.environ.get('OCR_LANGUAGE', 'ocrb_candidate')
if not re.fullmatch(r'[A-Za-z0-9_]+', OCR_LANGUAGE):
    raise ValueError('OCR_LANGUAGE 必須為模型名稱（不含路徑或副檔名）')

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
    """Locate tessdata directory containing the selected OCR model."""
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
        ocrb_file = os.path.join(candidate, OCR_LANGUAGE + ".traineddata")
        if os.path.exists(ocrb_file):
            logging.info(f"✅ 訓練數據文件已存在: {ocrb_file}")
            return candidate

    logging.error("找不到 OCR 模型 %s", OCR_LANGUAGE)
    raise FileNotFoundError(
        f"無法找到 {OCR_LANGUAGE}.traineddata，請設定 TESSDATA_PREFIX 或放入專案 tessdata 目錄。"
    )


tessdata_dir = resolve_tessdata_dir()
OCR_CONFIG = f'--tessdata-dir "{tessdata_dir}" --oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'


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
        if not isinstance(payload, dict):
            return jsonify({'status': 'error', 'message': 'JSON must be an object'}), 400
        base64_string = (
            payload.get('image_base64')
            or payload.get('image')
            or request.form.get('image_base64')
            or request.form.get('image')
        )

        if base64_string:
            if not isinstance(base64_string, str):
                return jsonify({'status': 'error', 'message': 'Invalid base64 image data'}), 400
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
        image = load_image(image_data)
        logging.info(f"🖼️ 影像來源: {source_desc} 已成功載入")

        data_page, mrz_data, detected_text = recognize_passport(image)
        
        # 中文辨識
        mrz_data['chinese_name'] = extract_chinese_name(data_page)

        return jsonify({'status': 'success', 'data': mrz_data}), 200

    except pytesseract.TesseractError:
        logging.exception('OCR engine error')
        return jsonify({'status': 'error', 'message': 'Internal Server Error'}), 500
    except RuntimeError:
        return jsonify({'status': 'error', 'message': 'OCR 執行逾時，請縮小圖片或重新拍攝'}), 504

    except ValueError as ve:
        logging.error(f"❌ MRZ 解析錯誤: {str(ve)}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400

    except Exception as e:
        logging.error(f"❌ OCR 過程發生錯誤: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Internal Server Error'}), 500


def load_image(data):
    """Decode input fully so invalid image errors are distinct from engine errors."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid image data') from exc


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
    if maxVal <= minVal:
        return Image.fromarray(gray)
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # 3. 閉合運算填補字元間隙
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    h, w = gray.shape

    # 4. 再次進行閉合與腐蝕，減少雜訊
    # 動態決定大核的大小，適應不同解析度
    k_w = int(w * 0.05) if w > 1000 else 21
    k_h = int(h * 0.05) if h > 1000 else 21
    bigKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, bigKernel)
    thresh = cv2.erode(thresh, None, iterations=2)

    # 5. 尋找輪廓
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    relevant_cnts = []
    
    for c in cnts:
        (x, y, cw, ch) = cv2.boundingRect(c)
        ar = cw / float(ch)
        crWidth = cw / float(w)

        # MRZ 通常寬度佔比 > 50%，高寬比 > 4 
        if ar > 4 and crWidth > 0.5:
            # 偏向影像下方 (護照 MRZ 位置)
            pY = y + (ch / 2)
            if pY > h * 0.4: # 放寬到下半部 60%
                candidates.append((x, y, cw, ch))
                relevant_cnts.append(c)

    if not candidates:
        logging.warning("⚠️ 無法精確偵測 MRZ 區域，使用預設比例裁剪 (下方 40%)")
        return Image.fromarray(gray[int(h * 0.6):, :])

    # 使用所有相關輪廓來計算帶角度的最小外接矩形 (minAreaRect)
    all_points = np.vstack(relevant_cnts)
    rect = cv2.minAreaRect(all_points)
    center, size, angle = rect
    
    # 計算角度 (OpenCV minAreaRect 的角度行為在不同版本可能不同)
    # 我們目標是把長度較長的一邊轉成水平
    if size[0] < size[1]:
        angle += 90
        size = (size[1], size[0])

    # Normalize the long-edge angle, avoiding upside-down crops.
    angle = (angle + 90) % 180 - 90
    # 執行旋轉校正
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # 在校正後的圖片中進行剪裁
    # 稍微放大 size 以包含 Padding
    cw, ch = size
    cw = int(cw * 1.05)
    ch = int(ch * 1.3) # 高度多給一點，確保兩行都在
    
    # 算出新的中心座標在旋轉後可能不變，直接從 center 裁切
    cx, cy = center
    x1 = max(0, int(cx - cw/2))
    y1 = max(0, int(cy - ch/2))
    x2 = min(w, int(cx + cw/2))
    y2 = min(h, int(cy + ch/2))
    
    cropped = rotated[y1:y2, x1:x2]
    logging.info(f"📐 執行旋轉剪裁: 角度 {angle:.2f}, 區域 ({x1},{y1}) -> ({x2},{y2})")
    
    # 如果裁切後太小， fallback
    if cropped.size == 0 or cropped.shape[0] < 20:
        return Image.fromarray(gray[int(h * 0.6):, :])

    # --- 影像優化 ---
    # 1. 若高度太小 (<60px)，等比例放大以符合 Tesseract 偏好的 30px 字高
    if cropped.shape[0] < 60:
        scale = 120.0 / max(cropped.shape[0], 1)
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. 強化對比度 (CLAHE)，平衡陰影與反光
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped = clahe.apply(cropped)
    
    # 3. 輕微降噪
    cropped = cv2.GaussianBlur(cropped, (3, 3), 0)

    return Image.fromarray(cropped)


def crop_to_data_page_via_face(image):
    """
    使用 OpenCV Haar Cascade 找尋人臉，並裁切掉上半部無關的頁面
    """
    cv_img = np.array(image)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    # 載入 Haar Cascade 模型
    if not hasattr(cv2, 'CascadeClassifier') or not hasattr(cv2, 'data'):
        logging.info('目前的 OpenCV 未提供人臉分類器，直接使用影像進行 MRZ 定位')
        return image
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        logging.warning("⚠️ 無法載入 haarcascade_frontalface_default.xml 模型")
        return image

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        # 取面積最大的臉
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # 護照照片通常在上方，我們保留這張臉以及其下方的所有內容
        # 往上延伸高度的 50% 避免切到頭頂
        crop_y_start = max(0, int(y - h * 0.5))
        logging.info(f"👤 偵測到最大人臉 (y:{y}, w:{w}, h:{h}), 裁切 Y 軸起點為 {crop_y_start}")
        
        # 如果人臉太靠下面，可能這張不是包含兩頁的照片，就不裁切
        if crop_y_start < gray.shape[0] * 0.7:
             cropped = cv_img[crop_y_start:, :]
             return Image.fromarray(cropped)
    
    logging.info("👤 未偵測到適合的人臉或不需裁切，使用原圖")
    return image


def preprocess_image_pil(image, include_original=False):
    """
    使用 PIL 和 OpenCV 處理影像，提高 OCR 的辨識精度
    """
    # 轉為灰階
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', image.size, 'white')
        image = Image.alpha_composite(background, rgba).convert('RGB')
    gray = ImageOps.grayscale(image)
    if gray.width < 20 or gray.height < 20:
        raise ValueError('圖片尺寸過小，無法辨識')
    if max(gray.size) > 2400:
        gray.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    
    # 1. 人臉輔助裁切 (去掉可能的上方簽名頁)
    cropped_by_face = crop_to_data_page_via_face(gray)
    
    # 2. 進行 MRZ 定位
    mrz_region = detect_mrz_region(cropped_by_face)
    if include_original:
        return cropped_by_face, mrz_region, gray
    return cropped_by_face, mrz_region


def mrz_image_variants(page, region, original=None):
    """Yield page/crop pairs lazily so successful scans avoid fallback processing."""
    yield page, region
    binary = Image.fromarray(cv2.threshold(
        np.array(region), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    yield page, binary
    # A large face on another ID card must not permanently remove the passport.
    original = original if original is not None else page
    yield original, original.crop((0, int(original.height * .45), original.width, original.height))
    yield original, original
    upper = original.crop((0, 0, original.width, int(original.height * .65)))
    yield original, detect_mrz_region(upper)
    inverted_page = original.transpose(Image.Transpose.ROTATE_180)
    yield inverted_page, detect_mrz_region(inverted_page)
    yield inverted_page, inverted_page


def recognition_attempts(page, region, original):
    """Try split lines and rectified strips before expensive full-page fallbacks."""
    variants = iter(mrz_image_variants(page, region, original))
    yield page, region, 'block', [(region, 6)]
    lines = split_mrz_lines(region)
    if lines:
        yield page, region, 'split_lines', [(line, 13) for line in lines]
        yield page, region, 'mixed_lines', [(lines[0], 13), (lines[1], 7)]
        yield page, region, 'single_lines', [(line, 7) for line in lines]
    next(variants)  # First region was already emitted above.
    oriented, binary = next(variants)
    yield oriented, binary, 'binary', [(binary, 6)]
    for candidate in perspective_mrz_candidates(original, limit=2):
        lines = split_mrz_lines(candidate)
        if lines:
            yield original, candidate, 'perspective_split', [(line, 13) for line in lines]
            yield original, candidate, 'perspective_mixed', [(lines[0], 13), (lines[1], 7)]
            yield original, candidate, 'perspective_single', [(line, 7) for line in lines]
        else:
            yield original, candidate, 'perspective_block', [(candidate, 6)]
    for oriented, variant in variants:
        yield oriented, variant, 'fallback', [(variant, 6)]


def recognize_passport(image, include_crop=False):
    """Read bounded block/line candidates without consulting training labels."""
    page, region, original = preprocess_image_pil(image, include_original=True)
    best = None
    best_score = None
    timed_out = 0
    calls = 0
    cache = {}
    attempts = 0
    deadline = time.monotonic() + 45
    for oriented_page, variant, method, inputs in recognition_attempts(page, region, original):
        # Keep source objects alive with the cache, so Python cannot reuse their ids.
        missing = sum((id(source), psm) not in cache for source, psm in inputs)
        if calls + missing > 14 or time.monotonic() >= deadline:
            break
        attempts += 1
        texts = []
        try:
            for source, psm in inputs:
                cache_key = (id(source), psm)
                if cache_key in cache:
                    texts.append(cache[cache_key][1])
                    continue
                original_source = source
                if psm == 6:
                    if source.width < 1400:
                        scale = min(3, 1400 / source.width)
                        source = source.resize((int(source.width * scale), int(source.height * scale)), Image.Resampling.LANCZOS)
                    source = ImageOps.expand(source, border=20, fill='white')
                # Raw-line normalization uses the whole line height. Extra block-style
                # padding would shrink the glyphs relative to the training images.
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('OCR request deadline')
                calls += 1
                recognized = pytesseract.image_to_string(
                    source, lang=OCR_LANGUAGE, config=OCR_CONFIG.replace('--psm 6', f'--psm {psm}'),
                    timeout=min(10, remaining))
                cache[cache_key] = (original_source, recognized)
                texts.append(recognized)
        except pytesseract.TesseractError:
            raise
        except RuntimeError:
            timed_out += 1
            continue
        text = '\n'.join(texts)
        try:
            data = extract_mrz_data(text)
        except ValueError:
            continue
        has_corrections = bool(data.get('corrections'))
        score = (data['verification']['overall'], not has_corrections,
                 sum(data['verification'].values()))
        if best_score is None or score > best_score:
            data['recognition'] = {
                'attempts': attempts, 'ocr_calls': calls, 'method': method,
                'needs_review': not data['verification']['overall'] or bool(data.get('corrections')),
            }
            best = (oriented_page, data, text, variant)
            best_score = score
        # A clean verified read takes precedence over a corrected suggestion.
        if data['verification']['overall'] and not has_corrections:
            return best if include_crop else best[:3]
    if best:
        best[1]['recognition'].update(attempts=attempts, ocr_calls=calls)
        return best if include_crop else best[:3]
    if timed_out or calls >= 14 or time.monotonic() >= deadline:
        raise RuntimeError('OCR timeout or request budget exhausted')
    raise ValueError('找不到完整護照 MRZ，請確認兩行機讀碼清晰、未被裁切或遮擋')


def extract_chinese_name(data_page_image):
    """
    從護照資料頁（去掉簽名頁與下方MRZ後）提取繁體中文姓名
    """
    cv_img = np.array(data_page_image)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    h, w = gray.shape[:2]
    # 我們只取上半部 60% 來找姓名，避免跟底下的其他英文資訊混在一起
    top_region = gray[:int(h * 0.6), :]
    
    # 增加對比度等簡單預處理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    top_region = clahe.apply(top_region)
    
    try:
        chi_dir = tessdata_dir if os.path.isfile(os.path.join(tessdata_dir, 'chi_tra.traineddata')) else DEFAULT_TESSDATA_DIR
        chi_config = f'--tessdata-dir "{chi_dir}" --psm 6'
        text = pytesseract.image_to_string(Image.fromarray(top_region), lang='chi_tra', config=chi_config, timeout=10)
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            line_no_space = re.sub(r'\s+', '', line)
            
            # 使用正則表達式尋找所有兩到四個字的純中文組合
            match = re.search(r'姓名.*?([\u4e00-\u9fa5]{2,4})', line_no_space)
            if match:
                return match.group(1)
        
        # 如果找不到 "姓名" 關鍵字，嘗試掃描所有行找尋合適的中文字串
        for line in lines:
            line_no_space = re.sub(r'\s+', '', line)
            if 2 <= len(line_no_space) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', line_no_space):
                 if line_no_space not in ['護照', '簽證', '外交部', '姓名']:
                     return line_no_space
                     
    except Exception as e:
        logging.error(f"❌ 中文姓名 OCR 失敗: {e}")
        
    return ""



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

        image = load_image(image_data)
        logging.info("🖼️ 影像已成功載入")

        # 預處理 + OCR
        data_page, mrz_data, detected_text = recognize_passport(image)

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

    except ValueError:
        return jsonify({'error': 'Invalid image or MRZ data'}), 400
    except pytesseract.TesseractError:
        return jsonify({'error': 'Internal Server Error'}), 500
    except RuntimeError:
        return jsonify({'error': 'OCR 執行逾時'}), 504
    except Exception as e:
        logging.error(f"❌ OCR CSV 處理過程發生錯誤: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    logging.info(f"🚀 啟動 Flask 伺服器，監聽 PORT {port}")
    app.run(host='0.0.0.0', port=port)
