import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Đọc dữ liệu từ file CSV
data  =pd.read_csv("./data.csv")

# Tạo cột dự đoán: 'Tien_dien' (giả sử đây là tổng công suất hoạt động nhân với cường độ)
data['Tien_dien'] = data['Cong_suat_hoat_dong_toan_cau'] * data['Cuong_do_toan_cau']

# Tách dữ liệu thành features và target
X = data[['Cong_suat_hoat_dong_toan_cau', 'Cong_suat_phan_khang_toan_cau', 'Dien_ap', 'Cuong_do_toan_cau']]
y = data['Tien_dien']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo các mô hình cơ sở (base models)
estimators = [
    ('lr', LinearRegression()),
    ('rd', Ridge()),
   
]


LinearRegression().fit(X_train,y_train)
Ridge().fit(X_train,y_train)

# Mô hình stacking với LinearRegression làm mô hình meta (final estimator)
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Huấn luyện mô hình stacking
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = stacking_model.predict(X_test)

# Đánh giá mô hình
def nash_sutcliffe_efficiency(obs, pred):
    return 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
nse = nash_sutcliffe_efficiency(y_test, y_pred)

# In kết quả
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"NSE: {nse:.4f}")