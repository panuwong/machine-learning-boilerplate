from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

features = [[120, 5], [50, 2], [230, 5], [20, 2], [60, 5], [59, 2]]
labels = ["eco car", "mocyc", "eco car", "mocyc", "eco car", "mocyc"]

# แปลง labels เป็นตัวเลขโดยใช้ One-Hot Encoding
encoder = LabelBinarizer()
encoded_labels = encoder.fit_transform(labels)

# สร้างโมเดล Logistic Regression และฝึกโมเดลด้วย features และ encoded_labels
reg = LogisticRegression()
reg.fit(features, encoded_labels)

# ทำนาย
prediction = reg.predict([[130, 2]])

# แปลงผลลัพธ์กลับเป็นข้อความ
decoded_prediction = encoder.inverse_transform(prediction)
print("ML = ",decoded_prediction)
