import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# matplotlib 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==== 读取数据 ====
file_path = r'E:\研三\研三上\论文\代码\Mymodel\data\2井数据.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(file_path)

df = pd.read_excel(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.dropna(inplace=True)

# ==== 构造序列 ====
def create_dataset(arr, time_step=1):
    X, y = [], []
    for i in range(len(arr)-time_step):
        X.append(arr[i:i+time_step])
        y.append(arr[i+time_step])
    return np.array(X), np.array(y)

time_step = 1
all_preds, mse_list, mae_list, rse_list = [], [], [], []

# ==== CNN 模型 ====
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32,1)
    def forward(self,x):
        # x: (batch, seq, 1) → (batch, 1, seq)
        x = x.transpose(1,2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

for seed in range(20):
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    X_train_np, y_train_np = create_dataset(train_df.iloc[:,0].values.astype(np.float32), time_step)
    X_test_np,  y_test_np  = create_dataset(test_df.iloc[:,0].values.astype(np.float32), time_step)

    X_train = torch.tensor(X_train_np[...,None], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np[...,None], dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test_np[...,None], dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_np[...,None], dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_train,y_train), batch_size=16, shuffle=True)

    model = CNNRegressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(50):
        for xb,yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
        actual = y_test.cpu().numpy().flatten()

    mse = mean_squared_error(actual, preds)
    mae = mean_absolute_error(actual, preds)
    rse = 1 - r2_score(actual, preds)
    mse_list.append(mse); mae_list.append(mae); rse_list.append(rse)

    df_pred = pd.DataFrame({
        'Date': test_df.index[time_step:],
        'Actual': actual,
        'Predicted': preds
    })
    all_preds.append(df_pred)
    print(f"Run {seed+1}: MSE={mse:.4f}, MAE={mae:.4f}, RSE={rse:.4f}")

print(f"\nAverage MSE: {np.mean(mse_list):.4f}, MAE: {np.mean(mae_list):.4f}, RSE: {np.mean(rse_list):.4f}")

# ==== 合并 & 绘图 ====
df_all = pd.concat(all_preds).sort_values('Date').reset_index(drop=True)
indices = np.linspace(0, len(df_all)-1, 200).astype(int)
df_plot = df_all.iloc[indices]

plt.figure(figsize=(12,6))
plt.plot(df_plot['Date'], df_plot['Actual'], label='真实值')
plt.plot(df_plot['Date'], df_plot['Predicted'], label='预测值')
plt.xlabel('日期'); plt.ylabel('产量'); plt.legend()
plt.show()
