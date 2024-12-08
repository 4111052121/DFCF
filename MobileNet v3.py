# ========== 模組引入 ==========
import os  # 用於處理檔案和目錄
from tensorflow.keras.applications import MobileNetV3Small  # 載入 MobileNetV3 小型模型
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # 用於添加全連接層
from tensorflow.keras.models import Model  # 用於構建自定義模型
from tensorflow.keras.optimizers import Adam  # 使用 Adam 優化器
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 圖像資料增強
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # 模型評估工具
import numpy as np  # 數值運算

# ========== 資料準備 ==========
# 訓練與測試資料的路徑
train_dir = "/mnt/external_drive_1/train"  # 訓練資料集目錄
test_dir = "/mnt/external_drive_2/test"  # 測試資料集目錄

# 類別定義
categories = ["real", "fake"]  # "real" 表示真實影像, "fake" 表示深偽影像

# ========== 資料增強與預處理 ==========
# 訓練集資料增強
train_datagen = ImageDataGenerator(
    rotation_range=30,  # 隨機旋轉角度範圍 (0~30 度)
    width_shift_range=0.2,  # 隨機水平平移比例
    height_shift_range=0.2,  # 隨機垂直平移比例
    brightness_range=[0.8, 1.2],  # 隨機亮度調整範圍
    zoom_range=0.2,  # 隨機縮放比例
    horizontal_flip=True,  # 隨機水平翻轉
    rescale=1.0 / 255  # 將像素值標準化到 [0,1] 範圍
)

# 測試集資料預處理
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)  # 僅標準化像素值

# 訓練資料生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 訓練資料所在目錄
    target_size=(224, 224),  # 圖像尺寸調整為 224x224
    batch_size=32,  # 每批次 32 張圖片
    class_mode='binary'  # 二分類模式
)

# 測試資料生成器
validation_generator = validation_datagen.flow_from_directory(
    test_dir,  # 測試資料所在目錄
    target_size=(224, 224),  # 圖像尺寸調整為 224x224
    batch_size=32,  # 每批次 32 張圖片
    class_mode='binary'  # 二分類模式
)

# ========== 模型構建 ==========
# 載入 MobileNetV3Small 作為基礎模型，不包含頂層
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定義分類層
x = base_model.output  # 取得基礎模型的輸出
x = GlobalAveragePooling2D()(x)  # 添加全局平均池化層
x = Dense(128, activation='relu')(x)  # 添加 128 神經元的全連接層，使用 ReLU 激活
predictions = Dense(1, activation='sigmoid')(x)  # 添加輸出層，Sigmoid 激活，二分類輸出

# 定義完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# ========== 模型訓練 ==========
# 1. 凍結基礎模型的參數，只訓練新增層
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練新增層
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # 訓練 10 個回合
    steps_per_epoch=len(train_generator),  # 每回合的步數
    validation_steps=len(validation_generator)  # 每回合的驗證步數
)

# 2. 啟用基礎模型的參數進行微調
for layer in base_model.layers:
    layer.trainable = True

# 使用較低學習率進行全模型微調
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練整個模型
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # 再訓練 10 個回合
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# ========== 模型評估 ==========
# 在測試集上評估模型
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 混淆矩陣與分類評估
y_true = validation_generator.classes  # 取得真實標籤
y_pred = (model.predict(validation_generator) > 0.5).astype(int)  # 預測結果 (閾值 0.5)
cm = confusion_matrix(y_true, y_pred)  # 計算混淆矩陣
print("Confusion Matrix:")
print(cm)

# 計算各種評估指標
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 顯示評估指標
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# ========== 儲存模型 ==========
model.save("deepfake_model_mobilenetv3.h5")  # 儲存模型為 H5 格式
print("Model has been saved.")
