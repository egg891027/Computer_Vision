import cv2
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque, Counter 
from tqdm import tqdm
import librosa
import numpy as np
import whisper
import moviepy as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import functional as F
import gc
import csv

# ==========================================
# 1. 全局設定
# ==========================================
VIDEO_PATH = './vlog.mp4'
MODEL_PATH_VISUAL = 'convnext_best.pth'
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, 'output_result.mp4')
OUTPUT_REPORT_PATH = os.path.join(RESULTS_DIR, 'final_analysis_integrated.csv') 
TEMP_AUDIO_PATH = 'temp_audio_final.wav'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

# HuggingFace 模型 ID
TEXT_MODEL_NAME = "Johnson8187/Chinese-Emotion"
AUDIO_MODEL_NAME = "xmj2002/hubert-base-ch-speech-emotion-recognition"

print(f"全局裝置設定: {DEVICE}")
print(f"結果將輸出至: {RESULTS_DIR}")

# 儲存每一幀的視覺情緒結果 [(timestamp, emotion, confidence), ...]
VISUAL_TIMELINE = []

# ==========================================
# 2. 視覺分析模組 (Visual Analysis)
# ==========================================
def build_convnext(num_classes):
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def run_visual_analysis():
    global VISUAL_TIMELINE
    VISUAL_TIMELINE = [] # 清空
    
    print("\n" + "="*50)
    print("階段一：視覺情緒分析 (Visual Analysis)")
    print("="*50)
    
    LABEL_MAP_CODE = {
        'a': 'neutral', 'b': 'happy', 'c': 'sad', 'd': 'angry',
        'e': 'disgust', 'f': 'fear', 'g': 'surprise'
    }
    CLASSES = list(LABEL_MAP_CODE.values())
    IDX_TO_CLASS = {i: label for i, label in enumerate(CLASSES)}
    
    print("正在載入 ConvNeXt 模型...")
    model = build_convnext(len(CLASSES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH_VISUAL, map_location=DEVICE))
        print("視覺模型載入成功！")
    except FileNotFoundError:
        print(f"找不到權重檔: {MODEL_PATH_VISUAL}，請確認路徑。")
        return

    model.to(DEVICE)
    model.eval()

    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    emotion_history = deque(maxlen=5)
    confidence_history = deque(maxlen=5)

    print(f"開始處理影片幀 (共 {total_frames} 幀)...")

    for frame_idx in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret: break
        
        # 計算當前時間戳記 (秒)
        current_timestamp = frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_frame_emotion = "no_face" 
        current_conf = 0.0

        if len(faces) > 0:
            target_face = max(faces, key=lambda f: f[2] * f[3])
            (x, y, w, h) = target_face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            try:
                face_img = frame[y:y+h, x:x+w]
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                input_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, preds = torch.max(probs, 1)
                    raw_emotion = IDX_TO_CLASS[preds.item()]
                    raw_conf = conf.item()

                emotion_history.append(raw_emotion)
                confidence_history.append(raw_conf)
                
                # 平滑化邏輯
                most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
                avg_conf = sum(confidence_history) / len(confidence_history)
                
                # 記錄到全域變數 (注意：這裡記錄平滑化後的結果)
                current_frame_emotion = most_common_emotion
                current_conf = avg_conf
                
                label_text = f"{most_common_emotion} ({avg_conf:.0%})"
                color = (0, 0, 255) if most_common_emotion in ['angry', 'fear', 'sad', 'disgust'] else (0, 255, 0)
                if most_common_emotion == 'neutral': color = (255, 255, 0)

                cv2.putText(frame, label_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            except:
                pass
        else:
            if len(emotion_history) > 0:
                emotion_history.clear()
                confidence_history.clear()
        
        # 修改點：加入 confidence
        VISUAL_TIMELINE.append((current_timestamp, current_frame_emotion, current_conf))
        
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"影片分析完成！已儲存至: {OUTPUT_VIDEO_PATH}")

# ==========================================
# 3. 多模態分析模組 (Audio & Text Analysis)
# ==========================================
class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HubertForSpeechEmotionRecognition(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()
    def forward(self, input_values, attention_mask=None):
        outputs = self.hubert(input_values, attention_mask=attention_mask)
        x = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(x)
        return SequenceClassifierOutput(logits=logits)

def get_dominant_visual_emotion(start_time, end_time):
    """
    從 VISUAL_TIMELINE 中找出該時間段出現最多次的情緒，
    並計算該情緒的平均信心度。
    回傳: (emotion_label, confidence_score)
    """
    # 篩選出該時段內有偵測到臉的數據
    relevant_data = [
        (emo, conf) for t, emo, conf in VISUAL_TIMELINE 
        if start_time <= t <= end_time and emo != "no_face"
    ]
    
    if not relevant_data:
        return "no_face", 0.0
    
    # 1. 找出出現最多次的情緒 (Mode)
    emotions = [d[0] for d in relevant_data]
    most_common_emotion = Counter(emotions).most_common(1)[0][0]
    
    # 2. 計算該情緒的平均信心度
    confs_of_dominant = [d[1] for d in relevant_data if d[0] == most_common_emotion]
    avg_conf = sum(confs_of_dominant) / len(confs_of_dominant)
    
    return most_common_emotion, avg_conf

def run_multimodal_analysis():
    print("\n" + "="*50)
    print("🎤 階段二：多模態整合分析 (Visual + Audio + Text)")
    print("="*50)

    if os.path.exists(TEMP_AUDIO_PATH): 
        try: os.remove(TEMP_AUDIO_PATH)
        except: pass
            
    video_clip = mp.VideoFileClip(VIDEO_PATH)
    video_clip.audio.write_audiofile(TEMP_AUDIO_PATH, logger=None)
    
    print("正在載入多模態模型 (Whisper, BERT, HuBERT)...")
    whisper_model = whisper.load_model("base", device=DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=False)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_model = HubertForSpeechEmotionRecognition.from_pretrained(
        AUDIO_MODEL_NAME, num_labels=6, ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    text_labels = {0: 'neutral', 1: 'concerned', 2: 'happy', 3: 'angry', 4: 'sad', 5: 'questioning', 6: 'surprise', 7: 'disgust'}
    audio_labels = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

    print("正在執行語音轉文字 (STT)...")
    result = whisper_model.transcribe(TEMP_AUDIO_PATH, language="zh")
    segments = result['segments']

    print(f"正在整合 Visual 數據並寫入 CSV...")
    
    with open(OUTPUT_REPORT_PATH, 'w', newline='', encoding='utf-8-sig') as csvfile:
        # 修改點：新增 Visual_Confidence 欄位
        fieldnames = [
            'Start_Time', 'End_Time', 'Content', 
            'Visual_Emotion', 'Visual_Confidence', 
            'Text_Emotion', 'Text_Confidence', 
            'Audio_Emotion', 'Audio_Confidence'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 終端機 Header
        print(f"{'Time':<12} | {'Visual':<15} | {'Text':<15} | {'Audio':<15} | {'Content'}")
        print("-" * 85)

        for seg in segments:
            start, end, text = seg['start'], seg['end'], seg['text']
            
            # 1. Visual Emotion (整合!)
            v_emo_label, v_conf_val = get_dominant_visual_emotion(start, end)
            
            # 2. Text Emotion
            t_emo_label = "neutral"
            t_conf_val = 0.0
            if text.strip():
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    t_out = text_model(**inputs)
                    t_prob = F.softmax(t_out.logits, dim=1)
                    t_idx = torch.argmax(t_prob).item()
                    t_emo_label = text_labels.get(t_idx, 'unknown')
                    t_conf_val = t_prob[0][t_idx].item()
            
            # 3. Audio Emotion
            a_emo_label = "neutral"
            a_conf_val = 0.0
            duration = end - start
            if duration >= 0.5:
                try:
                    y, sr = librosa.load(TEMP_AUDIO_PATH, sr=16000, offset=start, duration=duration)
                    if len(y) >= 16000:
                        a_inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
                        a_inputs = a_inputs.input_values.to(DEVICE)
                        with torch.no_grad():
                            a_out = audio_model(a_inputs)
                            a_prob = F.softmax(a_out.logits, dim=1)
                            a_idx = torch.argmax(a_prob).item()
                            a_emo_label = audio_labels.get(a_idx, 'unknown')
                            a_conf_val = a_prob[0][a_idx].item()
                except:
                    pass

            # 終端機顯示 (加上 Visual Confidence)
            v_display = f"{v_emo_label} ({v_conf_val:.0%})"
            t_display = f"{t_emo_label} ({t_conf_val:.0%})"
            a_display = f"{a_emo_label} ({a_conf_val:.0%})"

            print(f"{start:.1f}-{end:.1f}s".ljust(12) + " | " + 
                  f"{v_display}".ljust(15) + " | " + 
                  f"{t_display}".ljust(15) + " | " + 
                  f"{a_display}".ljust(15) + " | " +
                  (text[:10] + '..'))

            # 寫入 CSV
            writer.writerow({
                'Start_Time': round(start, 2),
                'End_Time': round(end, 2),
                'Content': text,
                'Visual_Emotion': v_emo_label,
                'Visual_Confidence': round(v_conf_val, 4), # 新增寫入
                'Text_Emotion': t_emo_label,
                'Text_Confidence': round(t_conf_val, 4),
                'Audio_Emotion': a_emo_label,
                'Audio_Confidence': round(a_conf_val, 4)
            })
    
    if os.path.exists(TEMP_AUDIO_PATH): 
        try: os.remove(TEMP_AUDIO_PATH)
        except: pass
    print(f"\n分析報告已儲存至: {OUTPUT_REPORT_PATH}")

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"錯誤: 找不到影片 {VIDEO_PATH}")
        exit()
    
    try:
        run_visual_analysis()     # 1. 跑視覺 (順便記錄每一幀的情緒與信心度)
        run_multimodal_analysis() # 2. 跑聲音文字 (整合所有信心度)
        
        print("\n全部作業流程完成！")
        print(f"1. 影片結果: {OUTPUT_VIDEO_PATH}")
        print(f"2. CSV 整合報告: {OUTPUT_REPORT_PATH}")
        
    except KeyboardInterrupt:
        print("\n使用者中斷程式。")
    except Exception as e:
        print(f"\n發生未預期的錯誤: {e}")