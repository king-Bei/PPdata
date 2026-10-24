# 預設護照 MRZ 模型

`ocrb_candidate.traineddata` 是以本目錄原版 `ocrb.traineddata` 微調的 Tesseract LSTM 模型，於 2026-10-25 啟用。使用已人工核對的台灣 TD3 護照 MRZ 單行標籤訓練，800 次迭代，學習率 0.0001。原始圖片、標籤及含個資的評估報告不隨版本庫提供。

- 模型 SHA-256：`39dc8697a6f35ebb77e47e3db8da9ed51d2deda1b6a5fda9d0ee1915d0b10955`
- 訓練／驗證／單行測試依護照分組，分別為 63／5／6 本護照，共 130／10／12 行。
- 初次獨立單行測試使用 PSM 13：原版 CER 4.55%、完整行 7/12；微調模型 CER 0.19%、完整行 11/12。此結果不代表整張護照準確率。
- 最終影像流程的 14 張開發回測中，原版與微調模型皆有 10 張全欄位符合人工標籤。這批照片已用來調整流程，不再視為獨立測試。另 24 張曾正確辨識的照片回歸檢查維持全欄位正確。

預設模型名稱為 `ocrb_candidate`；設定 `OCR_LANGUAGE=ocrb` 後重新啟動，可改用保留的原版模型。Docker 與 Compose 皆直接使用 `tessdata/` 內的模型。訓練、評估及資料限制請見 [OCR_TRAINING.md](../OCR_TRAINING.md)。
