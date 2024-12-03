import os  # 引入os模組，用於處理文件和目錄
import shutil  # 引入shutil模組，用於複製文件或資料夾
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 引入ImageDataGenerator進行資料增強
from tensorflow.keras.applications import Xception  # 引入Xception模型
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # 引入層進行模型構建
from tensorflow.keras.models import Model  # 引入Model類進行模型定義
from tensorflow.keras.optimizers import Adam  # 引入Adam優化器
from sklearn.metrics import confusion_matrix  # 引入混淆矩陣計算
import numpy as np  # 引入numpy

# ========== 資料準備 ==========
# 定義資料集路徑
train_dir = "/mnt/external_drive_1/train"  # 外接硬碟 1 上的訓練集資料夾
test_dir = "/mnt/external_drive_2/test"  # 外接硬碟 2 上的測試集資料夾
categories = ["real", "fake"]  # 類別，"real"表示真實影像，"fake"表示深偽影像

# ========== 資料增強與預處理 ==========
# 訓練集資料增強設置
train_datagen = ImageDataGenerator(
    rotation_range=30,  # 隨機旋轉範圍（0到30度）
    width_shift_range=0.2,  # 隨機水平平移範圍
    height_shift_range=0.2,  # 隨機垂直平移範圍
    brightness_range=[0.8, 1.2],  # 隨機亮度範圍
    zoom_range=0.2,  # 隨機縮放範圍
    horizontal_flip=True,  # 隨機水平翻轉
    rescale=1.0/255  # 圖像像素值歸一化到[0,1]範圍
)

# 測試集資料預處理設置（只做標準化）
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# 訓練資料生成器，會從外接硬碟中的訓練資料夾讀取圖片並進行增強
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 訓練集資料夾（外接硬碟 1）
    target_size=(224, 224),  # 圖像大小調整為224x224
    batch_size=32,  # 每次批次處理32張圖片
    class_mode='binary'  # 設定為二分類模式
)

# 測試資料生成器，只進行標準化處理
validation_generator = validation_datagen.flow_from_directory(
    test_dir,  # 測試集資料夾（外接硬碟 2）
    target_size=(224, 224),  # 圖像大小調整為224x224
    batch_size=32,  # 每次批次處理32張圖片
    class_mode='binary'  # 設定為二分類模式
)

# ========== 模型構建 ==========
# 加載預訓練的 Xception 模型，不包括頂層（因為我們需要自己定義）
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定義分類層
x = base_model.output  # 取得Xception模型的輸出
x = GlobalAveragePooling2D()(x)  # 全局平均池化層
x = Dense(128, activation='relu')(x)  # 全連接層，128個神經元，ReLU激活
predictions = Dense(1, activation='sigmoid')(x)  # 最後的輸出層，Sigmoid激活，二分類

# 構建模型
model = Model(inputs=base_model.input, outputs=predictions)

# ========== 模型訓練 ==========
# 1. 凍結預訓練層，只訓練新增的分類層
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型，使用Adam優化器，損失函數為二元交叉熵，評估指標為準確率
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練分類層
model.fit(
    train_generator,  # 訓練資料生成器
    validation_data=validation_generator,  # 測試資料生成器
    epochs=10,  # 訓練10個回合
    steps_per_epoch=len(train_generator),  # 每個回合的步數，根據訓練資料集大小確定
    validation_steps=len(validation_generator)  # 每個回合的驗證步數
)

# 2. 啟用全模型微調，對預訓練層進行微調
for layer in base_model.layers:
    layer.trainable = True  # 解凍所有層

# 使用較低的學習率進行微調
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 繼續訓練模型
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# ========== 模型評估 ==========
# 在測試集上進行評估
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 混淆矩陣，評估模型的分類性能
y_true = validation_generator.classes  # 真實標籤
y_pred = (model.predict(validation_generator) > 0.5).astype(int)  # 預測結果，閾值0.5進行分類
cm = confusion_matrix(y_true, y_pred)  # 計算混淆矩陣
print("Confusion Matrix:")
print(cm)

# 儲存模型
model.save("deepfake_model.h5")  # 儲存為 H5 格式的檔案
print("Model has been saved.")
