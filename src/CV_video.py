import cv2
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque, Counter 
from tqdm import tqdm

# ==========================================
# 1. 設定參數與路徑
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'convnext_best.pth'
VIDEO_PATH = './vlog.mp4'
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'output_result.mp4')
print(OUTPUT_PATH)

# 參數設定
WINDOW_SIZE = 12  # 平滑窗口：看過去 12 張圖來投票 (數字越大越穩，但反應越慢)
emotion_history = deque(maxlen=WINDOW_SIZE) # 儲存最近 N 次的情緒
confidence_history = deque(maxlen=WINDOW_SIZE) # 儲存最近 N 次的信心度

LABEL_MAP_CODE = {
    'a': 'neutral', 'b': 'happy', 'c': 'sad', 'd': 'angry',
    'e': 'disgust', 'f': 'fear', 'g': 'surprise'
}
CLASSES = list(LABEL_MAP_CODE.values())
IDX_TO_CLASS = {i: label for i, label in enumerate(CLASSES)}

# ==========================================
# 2. 載入模型
# ==========================================
def build_convnext(num_classes):
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

print("正在載入模型...")
model = build_convnext(len(CLASSES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ 模型載入成功！")
except FileNotFoundError:
    print(f"❌ 找不到權重檔: {MODEL_PATH}")
    exit()

model.to(DEVICE)
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 3. 處理影片主程序
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ 無法開啟影片: {VIDEO_PATH}，請確認路徑是否正確。")
    exit()

# 取得影片資訊 (寬、高、FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片資訊: {width}x{height}, FPS: {fps}, 總幀數: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"❌ 無法建立影片檔案: {OUTPUT_PATH}，請檢查路徑或權限。")
    cap.release()
    cv2.destroyAllWindows()
    exit()
print("🚀 開始分析影片... (這可能需要一點時間)")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # 找出面積最大的臉 (w * h)
        target_face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = target_face
        
        # 畫框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        try:
            # 預測
            face_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            input_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
                
                current_emotion = IDX_TO_CLASS[preds.item()]
                current_conf = conf.item()

            # 平滑化處理邏輯
            emotion_history.append(current_emotion)
            confidence_history.append(current_conf)
            
            # 1. 投票決定顯示哪個情緒 (Mode)
            # 例如: [Happy, Happy, Neutral, Happy, Happy] -> 顯示 Happy
            most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
            avg_conf = sum(confidence_history) / len(confidence_history)
            label_text = f"{most_common_emotion} ({avg_conf:.0%})"
            color = (0, 255, 0)
            if most_common_emotion in ['angry', 'fear', 'sad', 'disgust']:
                color = (0, 0, 255) # 紅色
            elif most_common_emotion == 'neutral':
                color = (255, 255, 0) # 黃色
            
            cv2.putText(frame, label_text, (x, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
        except Exception:
            pass
    else:
        # 如果沒偵測到臉，清空歷史，以免下次一偵測到就顯示舊的
        if len(emotion_history) > 0:
            emotion_history.clear()
            confidence_history.clear()
    
    # 寫入處理後的畫面到新影片
    out.write(frame)

    cv2.imshow('Real-time Emotion (Smoothed)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ 分析完成！結果已儲存為: {os.path.join(RESULTS_DIR, 'output_result.mp4')}")