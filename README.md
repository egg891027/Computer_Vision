# Computer-Vision

Read me first before using this database

The information for the Image_Info.xls
Column A: file_name: The file name of the image.
Column B: Self_evaluate: The self-evaluated intensity by the performer.
Column C: Observer_Count: Number of participants that rated this image.
Column D: maxIntCategory: The emotion category of this image based on the greatest intensity rated by the participants. 1: happy; 2: sad; 3: Angry; 4: disgusted; 5: fearful; 6: surprised.  
Column E: maxInt: The rated intensity in "maxIntCategory".  
Column F: EntropyVal: The entropy (inter-participant variability) of this image. 
Column G-L: counterMax: Proportion of participants that rated the image in this category.
Column M-R: entropyVal: Entropy computed from "counterMax".		
Column S-X: intVal: averaged intensity given by the observer in this category

The filename naming convention 

The first 2 digits: Performers' ID (from 01 to 30).
The 3rd digit:  Performance type 1: Theatric performamce; 2: Ekman's FACS criteria; 3: Personal event.
The 4th digit:  View point: 1: Front-view; 2: 3/4-view; 3: Profile-view.
the 5th digit(English letter): Type of expression performed: a: Calm/Netural; b: Happy; c: Sad; d: Angry; e: Disgusted; f: Fearful; g: Surprised.
The last 2 digits: serial number.

# Emotion Recognition on Taiwanese Faces with ConvNeXt

## 📌 Project Overview (專案簡介)
本專案旨在解決 **通用模型 (General Model)** 在特定族群（台灣人臉）上的 **領域偏移 (Domain Shift)** 問題。

原生的 DeepFace 模型在台灣人臉資料集上僅有 **40%** 的準確率。透過引入 SOTA 模型 **ConvNeXt Base** 並採用 **One-Stage 全解凍訓練 (Full Unfreeze)** 策略，我們成功將準確率提升至 **98.37%**，證明了針對特定場景進行微調 (Fine-tuning) 的必要性。

## 🚀 Key Features (技術亮點)
* **SOTA Model**: 使用 **ConvNeXt Base** 取代傳統的 ResNet/VGG，具備更強的特徵提取能力。
* **Training Strategy**: 
    * **One-Stage Training**: 不凍結骨幹，全網路參數同步更新，讓大模型完全適應小資料集。
    * **Cosine Annealing**: 使用餘弦退火調整學習率 (1e-4 $\to$ 1e-6)，精確收斂。
    * **Strong Regularization**: 設定 `Weight Decay = 0.05` 與 `RandomErasing`，有效防止過擬合 (Overfitting)。
* **Robustness**: 在驗證集上達到 **98.37% Accuracy**，大幅改善了 Fear (恐懼) 的辨識率。

## 📂 Project Structure (檔案結構)
```text
.
├── requirements.txt     # 依賴套件清單
├── convnext_best.pth    # 訓練好的最佳模型權重 (Accuracy: 98.37%)
├── README.md            # 專案說明文件
├── src/                 # 原始碼
│   ├── CV_image.py      # 模型訓練與靜態圖片評估 (含錯誤分析繪圖)
│   ├── CV_video.py      # 影片檔案分析 (針對 vlog.mp4)
│   └── live_demo.py     # Webcam 即時情緒偵測 (含防閃爍機制)
└── results/             # 分析結果圖表
    ├── confusion_matrix.png
    ├── loss_curve.png
    ├── error_analysis.png
    └── vlog_output.mp4

⚙️ Installation (安裝教學)
建議使用 Python 3.8+ 環境：

Bash

# 安裝必要套件
pip install -r requirements.txt
(註：若有 GPU，請確保 PyTorch 版本支援 CUDA 以加速訓練)

💻 Usage (使用說明)
1. 訓練與評估 (Training & Evaluation)
執行此指令可重新訓練模型，或載入 convnext_best.pth 產生混淆矩陣與錯誤分析圖。結果將自動儲存至 results/。

Bash

python src/CV_image.py
2. 影片分析 (Video Analysis)
針對指定的影片檔（如郭婞淳訪談 vlog.mp4）進行情緒分析。

Bash

python src/CV_video.py
輸入：預設讀取根目錄下的 vlog.mp4。

輸出：分析後的影片將存為 results/vlog_output.mp4。

3. 即時偵測 (Live Demo)
啟動 Webcam 進行即時情緒辨識，包含防閃爍 (Temporal Smoothing) 功能。

Bash

python src/live_demo.py
操作：按 q 鍵離開。

📊 Results & Analysis (成果分析)
1. Model Performance
Accuracy: 98.37% (大幅優於 Baseline 40%)

Loss Curve: 訓練 Loss 與驗證 Loss 同步下降，未出現明顯過擬合，證明 Weight Decay 策略有效。

Confusion Matrix: 在 Happy, Sad, Angry 等類別達到近乎 100% 的辨識率；Fear 的 Recall 提升至 0.94。

2. Case Study: Tears of Joy (郭婞淳影片分析)
在分析 vlog.mp4 時，模型傾向將 「喜極而泣」 的表情判讀為 Disgust (厭惡) 或 Sad (悲傷)。

觀察現象：

模型對於說話時的鼻部皺縮特徵非常敏感，容易將其歸類為 Disgust。

當淚水與悲傷特徵強烈時，快樂特徵被掩蓋。

原因分析：

視覺特徵重疊：說話時的肌肉牽動與強忍淚水的表情，在幾何特徵上與 Disgust 高度相似。

單一標籤限制 (Single-Label)：現有 Cross-Entropy 分類器無法處理 複合情緒 (Compound Emotions)，導致模型無法同時輸出 Happy + Sad。

結論：這顯示了從靜態圖片遷移至動態真實場景 (In-the-wild) 時的挑戰，未來可引入多模態 (Multimodal) 分析來解決此問題。