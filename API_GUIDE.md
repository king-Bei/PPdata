# 護照 OCR API 串接指南

本指南適用於提供 `X-API-Key` 驗證的版本。以下以本機 `http://localhost:5002` 為例；串接正式服務時，請替換為實際 HTTPS 網址。示例中的 `YOUR_API_KEY` 是佔位文字。

## 1. 管理與取得金鑰

1. 開啟 `/login`，使用管理帳號登入，進入首頁的「來源 App 金鑰」區塊。
2. 填寫來源名稱、App 名稱，以及永久或週期性金鑰。週期性金鑰需填起訖日期。
3. 按「儲存 API 設定」。伺服器會自動產生金鑰，不需手動輸入。
4. 複製清單或編輯欄位中的金鑰，放入串接系統的伺服器端環境變數。不要寫入公開前端程式、網址或版本庫。

按「重新產生」並確認後，新金鑰會立即儲存，舊金鑰無法再通過驗證；請同步更新串接端設定。重新產生保留原本的啟用狀態與有效日期，不會將已停用或已到期的設定自動啟用。

按「停用」後，該金鑰的新請求回傳 403；可按「啟用」恢復。週期性金鑰以 **Asia/Taipei 日期**判斷，包含起日及迄日；未到起日或超過迄日皆回傳 403。已進入辨識程序的請求仍會繼續完成。

已登入的管理頁可以用登入 Cookie 測試辨識。外部系統應使用 API 金鑰，不需先呼叫管理員登入 API。若請求明確提供 `X-API-Key`，一定檢查該金鑰，不會因同時有管理員 Cookie 而忽略停用狀態。

## 2. 上傳圖片辨識

- 方法：`POST /api/ocr`
- 驗證標頭：`X-API-Key: YOUR_API_KEY`
- 本文：`multipart/form-data`，檔案欄位名稱為 `image`。

```bash
export PP_OCR_BASE_URL='http://localhost:5002'
# PP_OCR_API_KEY 請在執行環境安全設定為管理頁取得的金鑰。
curl --fail-with-body "$PP_OCR_BASE_URL/api/ocr" \
  -H "X-API-Key: $PP_OCR_API_KEY" \
  -F 'image=@passport.jpg'
```

讓 curl 或 HTTP 函式庫自動設定 multipart 的 Content-Type 與 boundary，不要手動寫成 application/json。

### Base64 圖片

同一端點也接受 JSON 欄位 `image_base64` 或 `image`。可傳純 Base64，或 `data:image/jpeg;base64,...` 形式。

```bash
curl --fail-with-body "$PP_OCR_BASE_URL/api/ocr" \
  -H "X-API-Key: $PP_OCR_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

`request.json` 內容如下，請以完整圖片編碼取代佔位文字：

```json
{"image_base64": "BASE64_IMAGE_CONTENT"}
```

### Python 範例

此範例使用 `requests`（串接端需自行安裝）。

```python
import os
import requests

base_url = os.environ['PP_OCR_BASE_URL'].rstrip('/')
headers = {'X-API-Key': os.environ['PP_OCR_API_KEY']}
with open('passport.jpg', 'rb') as image:
    response = requests.post(
        base_url + '/api/ocr',
        headers=headers,
        files={'image': ('passport.jpg', image, 'image/jpeg')},
        timeout=(10, 90),
    )
response.raise_for_status()
result = response.json()['data']
needs_review = (not result['verification']['overall']
                or result['recognition']['needs_review'])
print('需要人工核對' if needs_review else 'MRZ 校驗通過')
```

處理時間依圖片品質與重試次數而異。OCR 階段最多 14 次引擎呼叫、共用 45 秒期限，另有前置影像處理與中文姓名辨識（上限 10 秒）；串接端讀取逾時需預留時間。

## 3. 回傳資料與核對方式

成功回傳 HTTP 200、`status: "success"`，資料位於 `data`：

| 欄位 | 說明 |
| --- | --- |
| `last_name` / `first_name` | MRZ 英文姓／名 |
| `chinese_name` | 中文姓名，可能為空字串 |
| `passport_number` | 護照號碼 |
| `nationality` | 國籍代碼，例如 TWN |
| `birth_date` / `expiry_date` | YYYY/MM/DD；無效時為 Invalid Date |
| `gender` | M、F、<（未指定），無效字元為空字串 |
| `ID_number` | 選填資料區內容 |
| `verification` | 護照號碼、出生日期、到期日、選填資料、綜合碼及 overall 校驗 |
| `recognition` | attempts、ocr_calls、method、needs_review |
| `corrections` | 若有，表示曾移除外圍雜訊或提出字元修正，需人工核對 |

**HTTP 200 表示可解析，不保證所有欄位正確。** `verification.overall` 為 false 或 `recognition.needs_review` 為 true 時，應安排人工核對。英文姓名與中文姓名不受 MRZ 校驗碼保護，即使全部校驗通過仍需依業務需求確認。完整欄位範例見 [API 回傳格式](api_response_spec.md)。

拍攝時請保留完整兩行 MRZ、護照平整且文字清晰，避免反光、手指遮擋與截圖裁切。缺漏字元無法靠重新產生金鑰或校驗碼補回。

## 4. CSV 端點

`POST /api/ocr-csv` 使用相同的 `X-API-Key` 驗證，僅接受 multipart 的 `image` 檔案欄位。

```bash
curl --fail-with-body "$PP_OCR_BASE_URL/api/ocr-csv" \
  -H "X-API-Key: $PP_OCR_API_KEY" \
  -F 'image=@passport.jpg'
```

此端點將辨識文字追加到**服務主機** `/tmp/tmp_output/output_YYYY-MM-DD.csv`，回傳 JSON 的 `message` 與 `detected_text`，不直接下載 CSV，也不回傳 `/api/ocr` 的完整欄位。容器重建可能清除 `/tmp` 資料。

## 5. 錯誤處理

| HTTP 狀態 | 含義與處理 |
| --- | --- |
| 400 | 圖片、Base64、MRZ 或管理欄位不正確；檢查輸入或重拍 |
| 401 | 未提供金鑰、金鑰不正確或舊金鑰已被替換；管理端則為未登入 |
| 403 | 金鑰停用、尚未生效、已到期，或跨來源管理操作被拒絕 |
| 404 | 找不到指定的金鑰設定 |
| 500 | OCR 引擎或服務執行失敗 |
| 503 | 金鑰儲存無法讀寫或資料損壞；請管理員檢查儲存狀態 |
| 504 | OCR 逾時；縮小圖片、改善清晰度或重拍後重試 |

一般錯誤格式為 `{"status":"error","message":"錯誤說明"}`；CSV 端點的辨識錯誤部分使用 `{"error":"錯誤說明"}`。建議先依 HTTP 狀態分類，再顯示 message 或 error。不要對 401／403 持續自動重試。

## 6. 管理 API

以下端點皆需要管理員登入 Cookie，不能使用一般 API 金鑰管理其他金鑰。管理介面已包含這些操作。

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| GET | `/api/app-keys` | 取得清單（包含金鑰，請勿公開） |
| POST | `/api/app-keys` | 新增來源，伺服器自動產生金鑰，回傳 201 |
| PUT | `/api/app-keys/{id}` | 更新名稱、類型、日期或狀態 |
| POST | `/api/app-keys/{id}/regenerate` | 立即替換金鑰，舊金鑰失效 |

新增 JSON：

```json
{
  "source_name": "旅行社 A",
  "app_name": "報名系統",
  "key_type": "periodic",
  "start_date": "2026-11-01",
  "end_date": "2026-11-30",
  "status": "active"
}
```

`source_name`、`app_name` 為必填文字。省略 `key_type` 時使用 permanent；省略 status 時使用 active。permanent 不使用起訖日期。api_key 由伺服器產生，新增時不採用客戶端提供的值；更新時禁止修改此欄位，請改用 regenerate。

停用／啟用只需以 PUT 傳送 `{"status":"disabled"}`／`{"status":"active"}`。重新產生端點不需要本文。操作成功回傳 `{"status":"success","data":{...}}`，包含已儲存的設定與金鑰。舊版未儲存的 `GET /api/app-keys/generate` 已移除，請使用新增或重新產生端點。

## 7. 部署與儲存設定

- `APP_KEYS_FILE`：金鑰 JSON 的絕對路徑；未設定時為專案根目錄的 app_keys.json。父資料夾必須存在且可寫入。儲存會建立同名 `.lock` 檔，透過鎖定與原子替換避免同主機多程序寫入遺失。寫入失敗不會替換既有金鑰。
- 金鑰目前以明文儲存，供管理頁檢視；新寫入檔案權限為 0600。請保護磁碟與備份，不要上傳此檔案。Docker 建置已排除此檔案，新的容器從空清單開始，需自行新增或掛載既有儲存。
- Compose 已設定命名 volume 保存金鑰；一般重建容器會保留，`docker compose down -v` 會刪除 volume。
- `SESSION_SECRET`：建議設定固定的隨機密值，讓重新啟動或多 worker 的管理登入 Cookie 保持一致。未設定時，每次程序啟動會隨機產生，重啟後需要重新登入。
- 本版為本機 JSON 金鑰儲存。Cloud Run 等臨時檔案系統不保證跨重啟保存，也無法在不同執行個體之間共享停用／重新產生結果。若正式環境使用多執行個體，需先改用共用資料庫，再依其交易機制同步金鑰狀態。
- 瀏覽器跨來源請求只開放程式內設定的來源；如有其他網站需直接呼叫，請先調整 CORS。從伺服器端串接不受瀏覽器 CORS 限制。
