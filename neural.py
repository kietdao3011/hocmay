import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu và chuẩn bị (thay df thành file của bạn)
df = pd.read_csv('data.csv')
df = df.replace('?', np.nan)
df.dropna(inplace=True)

# Chuẩn bị dữ liệu
x = df.drop(['mpg', 'car_name'], axis=1)
y = df['mpg']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Tạo mô hình MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', 
                   max_iter=1, warm_start=True, random_state=42)

# Số epoch
epochs = 150
train_loss_list = []
val_loss_list = []

for epoch in range(epochs):
    # Huấn luyện mô hình
    mlp.fit(x_train_scaled, y_train)
    
    # Dự đoán trên tập huấn luyện và kiểm tra
    y_train_pred = mlp.predict(x_train_scaled)
    y_test_pred = mlp.predict(x_test_scaled)
    
    # Tính toán cost (loss) cho tập huấn luyện và kiểm tra
    train_loss = mean_squared_error(y_train, y_train_pred)
    val_loss =  (y_test, y_test_pred)
    
    # Tính accuracy bằng R2 (cho bài toán regression)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_test, y_test_pred)
    
    # Lưu loss để in ra sau
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    
    # In ra thông tin cho từng epoch
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Training loss (MSE): {train_loss:.4f}, Validation loss (MSE): {val_loss:.4f}")
    print(f"  Training R^2: {train_r2:.4f}, Validation R^2: {val_r2:.4f}")

# Plot loss qua từng epoch
import matplotlib.pyplot as plt

plt.plot(range(1, epochs + 1), train_loss_list, label='Training Loss')
plt.plot(range(1, epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('MLP Training vs Validation Loss')
plt.legend()
plt.show()