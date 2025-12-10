import os
import glob
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from deepface import DeepFace
from tqdm import tqdm

# ==========================================
# 0. 設定隨機固定
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已設定隨機種子 Seed = {seed}")

# ==========================================
# 1. 參數設定
# ==========================================
DATA_DIR = './faces_256x256'
RESULTS_DIR = './results'
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
MODEL_SAVE_NAME = 'convnext_best.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP_CODE = {
    'a': 'neutral', 'b': 'happy', 'c': 'sad', 'd': 'angry',
    'e': 'disgust', 'f': 'fear', 'g': 'surprise'
}
CLASSES = list(LABEL_MAP_CODE.values())
CLASS_TO_IDX = {label: i for i, label in enumerate(CLASSES)}
IDX_TO_CLASS = {i: label for i, label in enumerate(CLASSES)}


class TaiwaneseFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"找不到資料夾: {os.path.abspath(root_dir)}")
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.data = []
        print(f"正在掃描資料夾: {root_dir} ...")
        for path in self.image_paths:
            filename = os.path.basename(path)
            try:
                if len(filename) > 4:
                    emotion_code = filename[4]
                    if emotion_code in LABEL_MAP_CODE:
                        label_idx = CLASS_TO_IDX[LABEL_MAP_CODE[emotion_code]]
                        self.data.append((path, label_idx))
            except: continue
        print(f"有效圖片數量: {len(self.data)}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label, path
        except Exception as e:
            return torch.zeros(3, 224, 224), label, path

def pipeline_predict(img_path, model):
    target_tensor = None
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_path, target_size=(224, 224), 
            detector_backend='mtcnn', enforce_detection=False, 
            align=True, grayscale=False
        )
        if len(face_objs) > 0:
            face_array = face_objs[0]['face'] 
            face_tensor = torch.tensor(face_array).permute(2, 0, 1).float()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            target_tensor = normalize(face_tensor).unsqueeze(0).to(DEVICE)
    except Exception: pass

    if target_tensor is None:
        try:
            img = Image.open(img_path).convert('RGB')
            manual_transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            target_tensor = manual_transform(img).unsqueeze(0).to(DEVICE)
        except: return "Error", 0.0

    with torch.no_grad():
        outputs = model(target_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        return IDX_TO_CLASS[preds.item()], conf.item()

# ==========================================
# 2. 建立模型
# ==========================================
def build_convnext(num_classes):
    print("建立 ConvNeXt 模型 (全解凍狀態)...")
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model.to(DEVICE)

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == '__main__':
    set_seed(42)
    torch.multiprocessing.freeze_support()
    print(f"使用裝置: {DEVICE}")

    # 維持您之前的強力增強配方
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        full_dataset_train = TaiwaneseFaceDataset(DATA_DIR, transform=train_transforms)
        full_dataset_val = TaiwaneseFaceDataset(DATA_DIR, transform=val_transforms)
    except FileNotFoundError as e:
        print(e)
        exit()

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size
    train_subset, _ = random_split(full_dataset_train, [train_size, val_size], generator=generator)
    _, val_subset = random_split(full_dataset_val, [train_size, val_size], generator=generator)

    # 建立兩個 DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 驗證用
    
    # 建立模型
    model = build_convnext(len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 紀錄歷史數據
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"\n=== 開始訓練 ({NUM_EPOCHS} epochs) ===")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, leave=True, unit="batch")
        for images, labels, _ in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
            loop.set_postfix(loss=loss.item(), acc=100*train_correct/train_total)
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step()

        # 紀錄數據
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"    Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2%}")
        print(f"    Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2%} (Best: {best_val_acc:.2%})")
        
        # 只存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            print(f"    🔥 發現新高分！模型已儲存至 {MODEL_SAVE_NAME}")

    print("訓練完成！")

    # 繪製趨勢圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss-accuracy_curve.png'))
    plt.show()

# ==========================================
# 4. 評估階段
# ==========================================
    print(f"\n=== 載入最佳模型 ({best_val_acc:.2%}) 進行最終整合評估 ===")
    model.load_state_dict(torch.load(MODEL_SAVE_NAME))
    model.eval()
    
    val_indices = val_subset.indices
    val_paths = [full_dataset_train.data[i][0] for i in val_indices]
    val_labels = [full_dataset_train.data[i][1] for i in val_indices]
    
    y_true, y_pred = [], []
    results = []
    
    for img_path, label_idx in tqdm(zip(val_paths, val_labels), total=len(val_paths)):
        true_label = IDX_TO_CLASS[label_idx]
        pred_label, conf = pipeline_predict(img_path, model)
        if pred_label == "Error": continue
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        results.append({
            "Filename": os.path.basename(img_path),
            "True": true_label,
            "Pred": pred_label,
            "Correct": true_label == pred_label,
            "Confidence": conf
        })

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n整體準確率: {accuracy:.2%}")
    print(classification_report(y_true, y_pred))

    df_res = pd.DataFrame(results)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.show()

# ==========================================
# 5. 錯誤案例分析
# ==========================================    
    wrong_cases = df_res[df_res['Correct'] == False]
    num_errors = len(wrong_cases)
    print(f"\n=== 預測錯誤案例分析 (共發現 {num_errors} 張錯誤) ===")
    
    if num_errors > 0:
        cols = 5
        rows = (num_errors // cols) + (1 if num_errors % cols > 0 else 0)
        plt.figure(figsize=(15, 3.5 * rows))
        
        for i, (_, row) in enumerate(wrong_cases.iterrows()):
            img_full_path = os.path.join(DATA_DIR, row['Filename'])
            
            if os.path.exists(img_full_path):
                img = cv2.imread(img_full_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img.shape
                    ax = plt.subplot(rows, cols, i+1)
                    plt.imshow(img)
                    
                    info_text = f"True: {row['True']}\nPred: {row['Pred']}\nConf: {row['Confidence']:.2%}"
                    if row['Confidence'] < 0.5:
                        text_color = 'red'   # 如果沒把握，設為紅色
                    else:
                        text_color = 'black' # 如果有把握，設為黑色
                    
                    plt.text(w + 10, 20, info_text, 
                             fontsize=11, color=text_color, 
                             va='top', ha='left', fontweight='bold')
                    plt.axis('off')
                    
            else:
                plt.subplot(rows, cols, i+1)
                plt.text(0.5, 0.5, "Image Not Found", ha='center', va='center')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'error_analysis.png'), bbox_inches='tight', dpi=300)
        print(f"正在顯示所有 {num_errors} 張錯誤圖片，請稍候...")
        plt.show()
        
    else:
        print("太完美了！測試集全部預測正確 (100% Accuracy)！")
        
# ==========================================
# 6. 成功案例展示
# ==========================================    
print("\n=== 正在儲存 success_cases.png===")

correct_cases = df_res[df_res['Correct'] == True]
unique_emotions = sorted(list(set(CLASSES)))
num_emotions = len(unique_emotions)
rows = 2
cols = 5
plt.figure(figsize=(20, 4 * rows))

for i, emotion in enumerate(unique_emotions):
    # 找出該情緒下，信心度最高的前 1 張
    best_case = correct_cases[correct_cases['True'] == emotion].sort_values(by='Confidence', ascending=False).head(1)
    
    if not best_case.empty:
        row = best_case.iloc[0]
        img_full_path = os.path.join(DATA_DIR, row['Filename'])
        
        if os.path.exists(img_full_path):
            img = cv2.imread(img_full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                
                # 建立子圖
                ax = plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                
                # --- 這裡使用您提供的邏輯 ---
                info_text = f"True: {row['True']}\nPred: {row['Pred']}\nConf: {row['Confidence']:.2%}"
                plt.text(w + 20, 20, info_text, 
                         fontsize=13, color='black', 
                         va='top', ha='left', fontweight='bold')
                
                plt.axis('off')

plt.subplots_adjust(wspace=1.2, hspace=0.3)
plt.suptitle(f"Top Success Cases per Emotion (Model Accuracy: {accuracy_score(y_true, y_pred):.2%})", fontsize=16, y=0.95)

save_path_success = os.path.join(RESULTS_DIR, 'success_cases.png')
plt.savefig(save_path_success, bbox_inches='tight', dpi=300)
plt.show()

print(f"已儲存成功案例圖: {save_path_success}")