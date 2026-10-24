# OCR 檢視、校正與重新訓練

目前辨識採用 OCR-B 預訓練模型，影像處理固定使用 OpenCV 4，以保留人臉輔助裁切介面。調整程式、儲存 OCR 結果或匯出 CSV 都不會自動更新模型權重。

## 已改善的辨識流程

- 修正 EXIF 方向、透明背景，限制送入影像處理的長邊為 2400 px。
- 結合定位區域、二值化、文字行配對與透視校正，再嘗試原圖區域、整頁及旋轉 180 度後的區域。分開讀取 MRZ 兩行時採用單行模式並保留訓練時的字高比例，避免整頁白邊使字元縮小。
- 每次請求最多 14 次 OCR，單次上限 10 秒，OCR 階段共用 45 秒期限（不含前置影像處理與另行執行的中文姓名 OCR）。同一行／同一模式會重用結果。無字元修正且校驗完整的結果優先；修正候選保留至其他方式讀取結束。
- 選用校驗結果最好的完整候選；校驗不全仍回傳可解析資料，並標記 `recognition.needs_review=true`。
- 支援兩行接在一起或換行切斷的 88 字元 MRZ，不補回看不見的護照字元或校驗碼。允許忽略姓名行末多出的填充符及數字行前方少量 OCR 邊界雜訊，但必須保留精確 44 字元且通過五項校驗。
- 護照號碼有單一易混淆字元時，只有一個替代候選能通過全部校驗才會提出修正，並回傳 `corrections` 與 `needs_review=true`，要求人工確認。姓名會在連續填充符開始處結束，尾端雜訊不會追加到名字。
- `verification.overall` 要求護照號碼、有效出生日期、有效到期日、選填資料、綜合碼共五項通過，並檢查性別與國籍欄位格式。
- 性別 `<` 保留為 `<`；其他無法辨識的內容回傳空字串並使整體驗證失敗，不再默認為女性。
- 出生日期選最近且不晚於今天的年份，到期日選最接近今天的世紀。世紀本身不在 MRZ 中，高齡或歷史護照仍需核對。
- 姓名沒有 MRZ 校驗碼。中文姓名仍使用原有獨立 OCR，需人工核對。
- API 不再把辨識出的護照文字寫入日誌；CSV 匯出仍依原功能儲存辨識文字。

## 資料準備

保留原始資料在受控位置。每張訓練圖片只放一行完整 MRZ，搭配同名 UTF-8 `.gt.txt`，內容必須是人工核對過的 44 個大寫字母、數字或 `<`。工具不會把 OCR 推測結果當成正確標籤。

```text
/private/path/ocr-data/
  train/
    passport001_line1.png
    passport001_line1.gt.txt
    passport001_line2.png
    passport001_line2.gt.txt
  validation/
    passport002_line1.png
    passport002_line1.gt.txt
  test/
    passport003_line1.png
    passport003_line1.gt.txt
```

依「護照／旅客」分組切分資料，例如 80% 訓練、10% 驗證、10% 最終測試。同一本護照的兩行、重拍圖、旋轉圖、增強版本都必須在同一組。工具會拒絕訓練與驗證集像素完全相同的圖片，但無法自動判斷不同圖片是否屬於同一本護照，分組需人工確認。

目前工具處理單行裁切圖。整張護照與兩行文字資料需先整理為以上配對格式；請另保留完整護照測試集，用於驗證區域偵測、解析及中文姓名。

## 執行重新訓練

安裝 Python 相依套件，並確認 PATH 中有 `tesseract`、`combine_tessdata`、`lstmtraining`。一般服務環境只有辨識引擎時，需另外安裝 Tesseract 訓練工具。

必須使用浮點模型 `tessdata/ocrb.traineddata`，不可使用已量化的 `ocrb_int.traineddata`。

```bash
python tools/ocr_training.py train \
  --train-dir /private/path/ocr-data/train \
  --eval-dir /private/path/ocr-data/validation \
  --base tessdata/ocrb.traineddata \
  --output training-runs/run-001 \
  --iterations 800
```

每次使用新的輸出資料夾。工具會建立 `.box`、`.lstmf`、分組清單、checkpoint、`ocrb_candidate.traineddata` 與樣本雜湊紀錄 `manifest.json`。不會覆蓋服務中的模型。

## 比較新舊模型

在未參與訓練或驗證的 `test` 資料上執行：

```bash
python tools/ocr_training.py evaluate \
  --data /private/path/ocr-data/test \
  --models tessdata/ocrb.traineddata training-runs/run-001/ocrb_candidate.traineddata \
  --report /tmp/ocr-comparison.json
```

報告包含字元錯誤率（CER，越低越好）及完整單行正確率（越高越好）。CER 包含漏字、插字及替換。報告不輸出護照文字；模型位置會顯示在報告內。

這是 OCR 單行評估，不能代表整張護照的辨識率。正式切換前，還要用完整護照測試 MRZ 定位、五項校驗通過率、各欄位正確率、誤判通過率及處理時間；以人工正確標籤比對，不能只用 Checksum 通過率宣稱準確率。

目前預設已啟用隨專案提供的 `tessdata/ocrb_candidate.traineddata`，重新啟動服務即可使用。原版 `ocrb.traineddata` 保留；需要還原時，明確設定 `OCR_LANGUAGE=ocrb` 並重新啟動。若服務原先已設定此環境變數，請先移除舊設定或改為 `ocrb_candidate`。後續訓練產生的新候選模型仍須評估後，再替換服務使用的模型。

專案內的 `training-data/` 與 `training-runs/` 預設已加入 `.gitignore`，並由 `.dockerignore` 排除於 Docker 建置內容。請勿把真實護照資料提交到版本庫。

訓練流程參考 [Tesseract 官方訓練文件](https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html) 與 [tesstrain 單行 box 格式](https://github.com/tesseract-ocr/tesstrain/blob/main/generate_line_box.py)。

## 程式驗證

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py mrz.py tools tests
```

測試涵蓋 ICAO 範例、綜合碼錯誤、選填碼錯誤、日期、性別、字元混淆、斷行／合併、候選排序、影像處理、OCR 重試／逾時、API 及訓練資料檢查。自動化測試以替代 OCR 回傳驗證流程；真實準確率與訓練結果需另用 Tesseract 及提供的資料實測。

## 從已核對的報名表整理照片

`tools/passport_dataset.py` 可讀取本專案使用的報名表欄位，將照片與正確資料配對。只能使用已人工核對的報名表；工具不執行工作簿公式或文件指令，也不修改原始檔案。

```bash
python tools/passport_dataset.py \
  --source /private/path/passports \
  --labels /private/path/verified-registration.xlsx \
  --tesseract /absolute/path/to/tesseract \
  --output training-runs/source-review
```

報名表數值欄位會處理科學記號與 Excel 日期。同名資料有不同護照號碼或欄位衝突時，會排除並記錄於 `label-conflicts.json`。資料整理用照片雜湊去重，以護照號碼的雜湊分組：尾碼分桶 0 為測試、1 為驗證、其餘為訓練。每本護照的所有照片與兩行文字使用同一組。

只有配對到完整正確欄位、辨識到台灣護照標頭，且圖像中的 MRZ 行與獨立標籤足夠接近時，才會輸出單行圖片及標籤。這個整理工具目前只產生台灣 TD3 護照的訓練配對，不適用其他國籍或證件種類。快取與模型／來源／程式版本綁定，條件變更請使用新的輸出資料夾。訓練資料生成使用 PSM 13，避免部分可見文字的行圖在一般分行模式下被忽略。

整理結果會保留部分裁切品質問題，應抽樣檢查。模型訓練與測試可直接使用輸出的 `dataset/train`、`dataset/validation`、`dataset/test`。

使用 `tools/evaluate_passports.py` 比較整張照片；其測試範圍包含未產生單行裁切圖的測試護照，辨識失敗也會計入正確率分母：

```bash
python tools/evaluate_passports.py \
  --source-review training-runs/source-review \
  --labels /private/path/verified-registration.xlsx \
  --tesseract /absolute/path/to/tesseract \
  --models tessdata/ocrb.traineddata training-runs/run-001/ocrb_candidate.traineddata \
  --report /tmp/full-passport-comparison.json
```

`OMP_THREAD_LIMIT` 預設設為 1，避免 OCR 執行緒過多而變慢；可在啟動前自行設定。人臉輔助裁切失敗時，後備辨識保留完整原圖，另嘗試原圖上方區域，以處理同張照片包含多個證件的情況。

## 開發回測與獨立測試的區別

只要照片已用於分析失敗原因或調整辨識流程，就改列開發回測，不再用它宣稱全新樣本準確率。`evaluate_passports.py` 可加 `--evaluation-role development` 記錄此用途。以前正確照片的回歸測試用於檢查退步，不能取代新的獨立測試。
