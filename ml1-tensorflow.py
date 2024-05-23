import tensorflow as tf
from tensorflow.keras import layers, models

# กำหนดข้อมูล features และ labels
features = tf.constant([[120, 5], [50, 2], [230, 5], [20, 2],[150, 5],[59, 2]], dtype=tf.float32)
labels = tf.constant([[1], [0], [1], [0], [1], [0]], dtype=tf.float32)  # 1 แทน "รถ" และ 0 แทน "มอไซค์"

# สร้างโมเดล Neural Network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),  # ใช้ activation function relu และมีจำนวนเซลล์ 64 ใน hidden layer แรก
    layers.Dense(64, activation='relu'),  # hidden layer ที่สอง
    layers.Dense(1)  # output layer ที่มีหนึ่งเซลล์ เนื่องจากเราต้องการคำตอบที่เป็นเลขเดียว
])

# คอมไพล์และคอนฟิกโมเดล
model.compile(optimizer='adam',  # เลือก optimizer วิธีหนึ่งในการปรับค่าน้ำหนักของโมเดล
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # เลือก loss function เนื่องจากเรามีสองคลาสในการจำแนก
              metrics=['accuracy'])  # ใช้ accuracy เพื่อวัดประสิทธิภาพของโมเดล

# ฝึกโมเดล
model.fit(features, labels, epochs=100)  # ฝึกโมเดลด้วยข้อมูล features และ labels ใน 100 epochs

# ทำนาย
prediction = model.predict(tf.constant([[140, 6]], dtype=tf.float32))  # ทำนายว่าข้อมูล [130, 1] เป็นรถหรือมอไซค์

# แสดงผลลัพธ์
if prediction >= 0.5:
    print("Prediction: Car",prediction)
else:
    print("Prediction: Motorcycle",prediction)
    
