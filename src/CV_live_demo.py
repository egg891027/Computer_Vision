import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import deque, Counter
import time

# ==========================================
# 1. 設定與參數
# ==========================================
# 自動抓取專案路徑 (不管在哪執行都能找到模型)
current_script_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(src_dir)
MODEL_PATH = os.path.join(project_root, 'convnext_best.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 5  # 平滑窗口：看過去 5 幀的結果來投票 (數值越大越穩，但反應越慢)

# 情緒標籤 (需與訓練時順序一致)
LABEL_MAP = ['neutral', 'happy', 'sad', 'angry', 'disgust', 'fear', 'surprise']

# ==========================================
# 2. 建立模型架構 (與訓練時相同)
# ==========================================
def build_convnext(num_classes):
    print("正在載入 ConvNeXt 模型架構...")
    # 這裡不需要預訓練權重，因為我們要載入自己的
    model = models.convnext_base(weights=None) 
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

# ==========================================
# 3. 初始化與載入權重
# ==========================================
print(f"使用裝置: {DEVICE}")
model = build_convnext(len(LABEL_MAP))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"成功載入模型權重: {MODEL_PATH}")
else:
    print(f"錯誤: 找不到權重檔 {MODEL_PATH}")
    print("請確認 convnext_best.pth 是否在專案根目錄中。")
    exit()

model.to(DEVICE)
model.eval()

# 定義影像前處理 (必須與訓練時一致)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 載入 OpenCV 的人臉偵測器 (Haar Cascade) - 速度快，適合即時應用
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 4. 定義平滑化類別
# ==========================================
class EmotionSmoother:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, emotion):
        self.history.append(emotion)

    def get_smoothed_emotion(self):
        if not self.history:
            return "Detecting..."
        # 找出出現次數最多的情緒 (Majority Voting)
        counts = Counter(self.history)
        most_common = counts.most_common(1)[0][0]
        return most_common

# 初始化平滑器
smoother = EmotionSmoother(window_size=WINDOW_SIZE)

# ==========================================
# 5. 開始即時偵測
# ==========================================
cap = cv2.VideoCapture('http://192.168.76.249:4747/video')  # 替換為你的 IP Camera URL 或使用 0 來使用內建攝影機

if not cap.isOpened():
    print("無法開啟攝影機，請檢查連接狀態。")
    exit()

print("\n=== 啟動即時情緒偵測 (按 'q' 離開) ===")
print(f"平滑化窗口大小: {WINDOW_SIZE} 幀")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 鏡像翻轉，讓畫面跟鏡子一樣
    frame = cv2.flip(frame, 1)
    
    # 轉灰階 (給 Haar Cascade 用)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # 計算 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    if len(faces) == 0:
        cv2.putText(frame, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    for (x, y, w, h) in faces:
        # 1. 畫出人臉框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        try:
            # 2. 擷取人臉區域 (ROI)
            face_roi = frame[y:y+h, x:x+w]
            
            # 3. 轉成 PIL 格式並進行前處理
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            input_tensor = val_transforms(face_pil).unsqueeze(0).to(DEVICE)

            # 4. 模型推論
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                raw_emotion = LABEL_MAP[pred.item()]
                confidence = conf.item()

            # 5. 平滑化處理 (投票機制)
            smoother.update(raw_emotion)
            final_emotion = smoother.get_smoothed_emotion()

            # 6. 顯示結果
            # 根據信心度決定顏色 ( >80% 綠色, <80% 黃色)
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            
            # 文字內容
            text_main = f"{final_emotion}"
            text_sub = f"Conf: {confidence:.2f}"

            # 放到畫面上
            cv2.putText(frame, text_main, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, text_sub, (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(f"Error processing face: {e}")
            continue

    # 顯示 FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Live Emotion Recognition (Press q to exit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()