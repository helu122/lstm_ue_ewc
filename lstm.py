import torch
import torch.nn as nn
import numpy as np


import matplotlib.pyplot as plt
# 伪造一段数据(正弦波数据）

def generate_sine_data(n_samples=1000,seq_len=100,noise_std = 0.05):
    np_stack = []
    for i in range(n_samples):
        fre = np.random.uniform(0.5,2.0)
        phase = np.random.uniform(0,2*np.pi)
        t = np.linspace(0,1,seq_len)
        x = np.sin(2*np.pi*fre*t+phase)
        noise  = np.random.normal(0,noise_std,x.shape)
        x += noise
        np_stack.append(x)
    data=np.stack(np_stack)
    data = data[...,np.newaxis]
    return data
# LSTM-AE模型

class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        # decoder — 注意：输入是 latent_dim，不是 input_dim
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        z = self.encoder_fc(h_n[-1])      # [batch, latent_dim]

        # Decode: 每一步输入 z，让 LSTM 自己展开时序
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, latent_dim]
        dec_out, _ = self.decoder_lstm(z_repeated)          # [batch, seq_len, hidden_dim]
        dec_out = self.output_fc(dec_out)                    # [batch, seq_len, input_dim]

        return dec_out

# ============================================
# 1. 生成数据
# ============================================
data = generate_sine_data(n_samples=1000, seq_len=100, noise_std=0.05)
data_tensor = torch.tensor(data, dtype=torch.float32)  # [1000, 100, 1]

# 划分训练/测试
split = int(0.8 * len(data_tensor))
train_data = data_tensor[:split]
test_data  = data_tensor[split:]

# ============================================
# 2. 初始化模型
# ============================================
input_dim  = 1
hidden_dim = 32
latent_dim = 8

model = LSTMAE(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================
# 3. 训练
# ============================================
epochs = 50
batch_size = 64

model.train()
for epoch in range(epochs):
    perm = torch.randperm(len(train_data))
    total_loss = 0
    for i in range(0, len(train_data), batch_size):
        idx = perm[i:i+batch_size]
        batch = train_data[idx]

        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)   # 重构误差
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ============================================
# 4. 可视化
# ============================================
model.eval()
with torch.no_grad():
    sample = test_data[:5]              # 取5条测试数据
    recon = model(sample)               # 模型重构
    sample_np = sample.numpy()
    recon_np  = recon.numpy()

fig, axes = plt.subplots(5, 1, figsize=(10, 8))
for i in range(5):
    axes[i].plot(sample_np[i, :, 0], label='Original', linewidth=1.5)
    axes[i].plot(recon_np[i, :, 0], label='Reconstructed', linestyle='--', linewidth=1.5)
    axes[i].fill_between(range(100), sample_np[i, :, 0], recon_np[i, :, 0], alpha=0.3, color='red')
    axes[i].set_ylabel(f'Sample {i+1}')
    if i == 0:
        axes[i].legend()
    if i == 4:
        axes[i].set_xlabel('Time Step')

plt.suptitle('LSTM-AE: Original vs Reconstructed (Sine Waves)')
plt.tight_layout()
plt.show()

# 重构误差分布
errors = ((sample_np - recon_np) ** 2).mean(axis=(1, 2))
print(f"\nReconstruction Errors (MSE) for 5 test samples: {errors}")
# ============================================
# 5. 频率敏感性测试（模拟"新地层"）
# ============================================
model.eval()
test_freqs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
errors_by_freq = []

with torch.no_grad():
    for f in test_freqs:
        # 生成该频率的 50 条正弦波
        t = np.linspace(0, 1, 100)
        samples = []
        for _ in range(50):
            phase = np.random.uniform(0, 2*np.pi)
            x = np.sin(2 * np.pi * f * t + phase)
            x += np.random.normal(0, 0.05, x.shape)
            samples.append(x)
        data_f = np.stack(samples)[..., np.newaxis]  # [50, 100, 1]
        data_t = torch.tensor(data_f, dtype=torch.float32)

        recon_f = model(data_t)
        mse = ((data_t - recon_f) ** 2).mean().item()
        errors_by_freq.append(mse)
        print(f"  Frequency {f:.1f} Hz → MSE: {mse:.4f}")

# 画 RE vs 频率曲线
plt.figure(figsize=(8, 4))
plt.plot(test_freqs, errors_by_freq, marker='o', linewidth=2, color='darkred')
plt.axvspan(0.5, 2.0, alpha=0.15, color='green', label='Training Range')
plt.axvspan(2.0, 8.0, alpha=0.15, color='red', label='Unseen Range')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('RE vs Signal Frequency — OOD Detection Simulation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
