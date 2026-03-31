# 護照 OCR API 資料格式與解析說明文件

本文件詳細說明護照 OCR (Optical Character Recognition) 介面回傳之資料結構、欄位定義以及後端解析邏輯。

---

## 1. 介面概覽

- **端點位置**: `/api/ocr`
- **請求方法**: `POST`
- **資料格式**: `multipart/form-data` (上傳檔案) 或 `application/json` (Base64 影像)
- **主要功能**: 讀取護照影像中的機讀區 (MRZ, Machine Readable Zone)，解析並回傳正確的旅客資訊。

---

## 2. API 回傳資料格式 (JSON)

當辨識成功時，API 將回傳狀態碼 `200 OK`，其內容結構如下：

```json
{
  "status": "success",
  "data": {
    "chinese_name": "王大明",
    "last_name": "WANG",
    "first_name": "DA MING",
    "passport_number": "310000000",
    "nationality": "TWN",
    "birth_date": "1990/01/01",
    "gender": "M",
    "expiry_date": "2030/12/31",
    "ID_number": "A123456789",
    "verification": {
      "passport": true,
      "birth_date": true,
      "expiry_date": true,
      "overall": true
    }
  }
}
```

---

## 3. 欄位定義說明

### 3.1 核心資料欄位 (`data`)

| 欄位名稱 | 類型 | 說明 | 範例 |
| :--- | :--- | :--- | :--- |
| `chinese_name` | String | 中文姓名 (若無則為空字串 `""`) | 王大明 |
| `last_name` | String | 姓氏 (大寫英文) | WANG |
| `first_name` | String | 名字 (大寫英文) | DA MING |
| `passport_number` | String | 護照號碼 | 310000000 |
| `nationality` | String | 國籍代碼 (ISO 3166-1 alpha-3) | TWN |
| `birth_date` | String | 出生日期 (格式：`YYYY/MM/DD`) | 1990/01/01 |
| `gender` | String | 性別 (`M`: 男, `F`: 女) | M |
| `expiry_date` | String | 效期截止日 (格式：`YYYY/MM/DD`) | 2030/12/31 |
| `ID_number` | String | 身分證字號或選填區域內容 | A123456789 |

### 3.2 驗證資訊 (`verification`)

系統會針對 MRZ 內的校驗碼進行檢查，確保資料讀取之正確性：

| 欄位名稱 | 類型 | 說明 |
| :--- | :--- | :--- |
| `passport` | Boolean | 護照號碼校驗結果 |
| `birth_date` | Boolean | 出生日期校驗結果 |
| `expiry_date` | Boolean | 效期截止日校驗結果 |
| `overall` | Boolean | 所有必要校驗是否全部通過 |

---

## 4. 解析邏輯流程

後端解析過程包含多個關鍵步驟，以確保在不同光視、角度下的影像皆能正確讀取：

### 4.1 影像預處理
1. **區域偵測 (MRZ Detection)**: 使用 OpenCV 形態學運算 (Morphology Ops) 定位護照下方的文字區塊。
2. **傾斜校正**: 自動計算文字傾斜角度並旋轉回正，確保文字成水平狀態。
3. **對比度強化 (CLAHE)**: 針對陰影或反光區域進行自適應長直圖均衡化，提升字元邊緣清晰度。

### 4.2 文字辨識 (OCR)
- 使用 **Tesseract OCR** 搭配專用的 `ocrb` 語言包（護照標準字體）。
- **字元白名單**: 限制辨識範圍為 `A-Z`, `0-9`, `<`，避免產生特殊符號干擾。

### 4.3 格式解析與校正
- **姓名解析**: 根據 `<<` 分隔符號拆分姓氏與名字。
- **自動容錯修復**: 針對 OCR 易混淆字元進行二次修正：
    - 日期欄位的 `O` 或 `D` 會被修正為 `0`。
    - 名字欄位的 `0` 會被修正為 `O`。
- **日期轉換**: 將 MRZ 的 `YYMMDD` 格式轉換為系統易讀的 `YYYY/MM/DD`。
- **校驗碼計算 (ICAO 9303)**:
    - 採用權重計算法 `[7, 3, 1]` 循環計算每個欄位的校驗位，並與 MRZ 內的校驗碼對比。

---

## 5. 常見錯誤碼

| 狀態碼 | 訊息範例 | 原因說明 |
| :--- | :--- | :--- |
| `400` | `No image data provided` | 請求中未包含有效的圖片檔案或 Base64 字串。 |
| `400` | `MRZ 資料不足，無法解析` | 圖片解析度過低或拍攝不全，導致無法完整讀取 MRZ 兩行文字。 |
| `500` | `Internal Server Error` | 伺服器端發生非預期錯誤（如：OCR 組件異常）。 |

---
> [!TIP]
> 為了獲得最佳辨識效果，建議拍攝時光線充足、護照平整，且確保下方的兩行機讀碼區域沒有被手指或反光遮擋。
