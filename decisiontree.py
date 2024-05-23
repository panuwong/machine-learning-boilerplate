from sklearn import tree

features = [[120, 5], [50, 2], [230, 5], [20, 2], [60, 5], [59, 2]]
labels = ["eco car", "mocyc", "eco car", "mocyc", "eco car", "mocyc"]


# สร้างโมเดล Logistic Regression และฝึกโมเดลด้วย features และ labels
reg = tree.DecisionTreeClassifier()
reg.fit(features, labels)

# ทำนาย
prediction = reg.predict([[130, 1]])

print("RS = ",prediction)
