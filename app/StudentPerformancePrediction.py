import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = {
    "Study_Hours": [2,4,6,8,10,1,3,5,7,9],
    "Attendance": [60,65,70,75,90,50,55,68,78,85],
    "Previous_Score": [50,55,65,70,80,45,48,60,72,78],
    "Final_Score": [52,58,68,74,85,48,50,63,76,82]
}

df = pd.DataFrame(data)
print(df)


print(df.describe())

plt.scatter(df["Study_Hours"], df["Final_Score"])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.show()

X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", list(y_pred))


print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [72],
    "Previous_Score": [66]
})

prediction = model.predict(new_student)
print("Predicted Final Score:", prediction[0])


new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [72],
    "Previous_Score": [66]
})

prediction = model.predict(new_student)
print("Predicted Final Score:", prediction[0])
