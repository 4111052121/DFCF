import os  # 用於處理文件和目錄操作
import numpy as np  # 用於數據處理
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 用於圖像增強和預處理
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D  # 用於構建全連接層和池化層
from tensorflow.keras.models import Model  # 用於定義 Keras 模型
from tensorflow.keras.optimizers import Adam  # 使用 Adam 優化器
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # 用於評估模型
from transformers import ViTModel, ViTFeatureExtractor  # 用於加載 Hugging Face 的 Vision Transformer 模型

# ======== 設定與資料路徑 =========
train_dir = "/mnt/external_drive_1/train"  # 訓練集資料夾路徑
test_dir = "/mnt/external_drive_2/test"  # 測試集資料夾路徑
categories = ["real", "fake"]  # 定義類別標籤

# ======== 資料增強與預處理 =========
# 定義訓練集的圖像增強策略
train_datagen = ImageDataGenerator(
    rotation_range=30,  # 隨機旋轉角度範圍
    width_shift_range=0.2,  # 隨機水平平移
    height_shift_range=0.2,  # 隨機垂直平移
    brightness_range=[0.8, 1.2],  # 隨機調整亮度
    zoom_range=0.2,  # 隨機縮放
    horizontal_flip=True,  # 隨機水平翻轉
    rescale=1.0 / 255  # 將像素值歸一化到 [0, 1]
)

# 定義測試集的圖像處理策略（僅做標準化）
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 建立訓練資料生成器，讀取並增強訓練集圖片
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 訓練集資料夾路徑
    target_size=(224, 224),  # 將圖像調整為 224x224
    batch_size=32,  # 每批次處理 32 張圖片
    class_mode='binary'  # 二分類標籤模式
)

# 建立測試資料生成器，讀取並標準化測試集圖片
validation_generator = validation_datagen.flow_from_directory(
    test_dir,  # 測試集資料夾路徑
    target_size=(224, 224),  # 將圖像調整為 224x224
    batch_size=32,  # 每批次處理 32 張圖片
    class_mode='binary'  # 二分類標籤模式
)

# ======== 加載 ViT-G 模型 =========
# 使用 Hugging Face 加載預訓練的 Vision Transformer 模型
vit_model = ViTModel.from_pretrained("google/vit-g-14")
vit_model.trainable = False  # 凍結預訓練模型的參數（第一階段不更新）

# ======== 建立自定義分類器 =========
# 定義模型的輸入層，輸入圖像尺寸為 224x224x3
input_layer = keras.Input(shape=(224, 224, 3))  

# 預處理輸入（選用 EfficientNetB0 作為例子）
preprocessed = EfficientNetB0(weights=None, include_top=False)(input_layer)

# 提取 ViT 的特徵，輸出序列化特徵
vit_features = vit_model(preprocessed)[0]  

# 全局平均池化，將序列特徵壓縮為一維特徵
pooled_features = GlobalAveragePooling1D()(vit_features)  

# 添加全連接層，128 個神經元，激活函數為 ReLU
dense_layer = Dense(128, activation="relu")(pooled_features)  

# 添加輸出層，1 個神經元，激活函數為 Sigmoid（用於二分類）
output_layer = Dense(1, activation="sigmoid")(dense_layer)  

# 定義整體模型，從輸入層到輸出層
model = Model(inputs=input_layer, outputs=output_layer)

# ======== 訓練分類層 =========
# 編譯模型，僅訓練分類層（凍結預訓練部分）
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# 訓練模型的分類層
model.fit(
    train_generator,  # 使用訓練集生成器
    validation_data=validation_generator,  # 使用測試集生成器
    epochs=10,  # 訓練 10 個回合
    steps_per_epoch=len(train_generator),  # 每個回合的步數
    validation_steps=len(validation_generator)  # 驗證的步數
)

# ======== 全模型微調 =========
# 解凍預訓練模型，允許所有參數進行微調
vit_model.trainable = True  

# 使用更低的學習率進行微調
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# 訓練整個模型（包括預訓練層）
model.fit(
    train_generator,  # 使用訓練集生成器
    validation_data=validation_generator,  # 使用測試集生成器
    epochs=10,  # 訓練 10 個回合
    steps_per_epoch=len(train_generator),  # 每個回合的步數
    validation_steps=len(validation_generator)  # 驗證的步數
)

# ======== 評估模型性能 =========
# 在測試集上評估模型
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 預測測試集的標籤
y_true = validation_generator.classes  # 真實標籤
y_pred = (model.predict(validation_generator) > 0.5).astype(int)  # 將預測值轉換為二分類標籤

# 計算並打印混淆矩陣
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 計算並打印各種評估指標
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# ======== 儲存模型 =========
# 將訓練好的模型保存為 H5 格式
model.save("vit_g_deepfake_model.h5")
print("Model has been saved.")
