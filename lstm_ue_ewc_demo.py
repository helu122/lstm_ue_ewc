import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 伪造一段数据(正弦波数据）
# ============================================
def generate_sine_data(n_samples=1000, seq_len=100, noise_std=0.05):
    np_stack = []
    for i in range(n_samples):
        fre = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, seq_len)
        x = np.sin(2 * np.pi * fre * t + phase)
        noise = np.random.normal(0, noise_std, x.shape)
        x += noise
        np_stack.append(x)
    data = np.stack(np_stack)
    data = data[..., np.newaxis]
    return data


# ============================================
# LSTM-AE 模型（已加 MC Dropout）
# ============================================
class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p

        # encoder — LSTM 自带 dropout 参数
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                                    dropout=dropout_p)       # ← LSTM 内部 dropout
        self.enc_drop = nn.Dropout(dropout_p)                # ← fc 前再一层
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True,
                                    dropout=dropout_p)       # ← LSTM 内部 dropout
        self.dec_drop = nn.Dropout(dropout_p)                # ← fc 前再一层
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        z = self.enc_drop(h_n[-1])                           # Dropout 在 fc 之前
        z = self.encoder_fc(z)                               # [batch, latent_dim]

        # Decode
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(z_repeated)
        dec_out = self.dec_drop(dec_out)                     # Dropout 在 fc 之前
        dec_out = self.output_fc(dec_out)

        return dec_out



# ============================================
# 1. 生成数据
# ============================================
data = generate_sine_data(n_samples=1000, seq_len=100, noise_std=0.05)
data_tensor = torch.tensor(data, dtype=torch.float32)  # [1000, 100, 1]

# 划分训练/测试
split = int(0.8 * len(data_tensor))
train_data = data_tensor[:split]
test_data = data_tensor[split:]

# ============================================
# 2. 初始化模型
# ============================================
input_dim = 1
hidden_dim = 32
latent_dim = 8

model = LSTMAE(input_dim, hidden_dim, latent_dim, dropout_p=0.2)
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
        idx = perm[i:i + batch_size]
        batch = train_data[idx]

        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)  # 重构误差
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# ============================================
# 4. 可视化（eval 模式：Dropout 关闭，确定性输出）
# ============================================
model.eval()
with torch.no_grad():
    sample = test_data[:5]  # 取5条测试数据
    recon = model(sample)  # 模型重构
    sample_np = sample.numpy()
    recon_np = recon.numpy()

fig, axes = plt.subplots(5, 1, figsize=(10, 8))
for i in range(5):
    axes[i].plot(sample_np[i, :, 0], label='Original', linewidth=1.5)
    axes[i].plot(recon_np[i, :, 0], label='Reconstructed', linestyle='--', linewidth=1.5)
    axes[i].fill_between(range(100), sample_np[i, :, 0], recon_np[i, :, 0], alpha=0.3, color='red')
    axes[i].set_ylabel(f'Sample {i + 1}')
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
        t = np.linspace(0, 1, 100)
        samples = []
        for _ in range(50):
            phase = np.random.uniform(0, 2 * np.pi)
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

# ============================================
# 6. MC Dropout 不确定性分解（核心）
# ============================================
K = 20  # MC Dropout 前向次数

# 生成 3 类测试样本：训练范围内 / 边界 / 未见范围
test_samples = {
    "In-Distribution (1.0 Hz)": 1.0,
    "Boundary (2.5 Hz)": 2.5,
    "OOD (5.0 Hz)": 5.0,
}

print("\n" + "=" * 60)
print("MC Dropout Uncertainty Decomposition (K=20)")
print("=" * 60)

# 关键：保持 model.train() 让 Dropout 在推理时激活
model.train()
with torch.no_grad():
    for label, freq in test_samples.items():
        t = np.linspace(0, 1, 100)
        phase = np.random.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * freq * t + phase)
        x += np.random.normal(0, 0.05, x.shape)
        x_tensor = torch.tensor(x.reshape(1, 100, 1), dtype=torch.float32)

        # K 次随机前向传播
        x_hats = []
        for k in range(K):
            x_hat = model(x_tensor)
            x_hats.append(x_hat)
        x_hats = torch.stack(x_hats, dim=0)          # [K, 1, 100, 1]
        x_bar = x_hats.mean(dim=0)                    # [1, 100, 1]

        # 不确定性分解
        RE_k = ((x_tensor - x_hats) ** 2).mean(dim=(2, 3))  # [K, 1]
        sigma_a = RE_k.mean().item()                  # 偶然不确定性（RE 均值）
        sigma_e = ((x_hats - x_bar) ** 2).mean().item()  # 认知不确定性（模型分歧）

        print(f"\n{label}:")
        print(f"  Aleatoric (σ²_a): {sigma_a:.6f}  ← 数据噪声/固有RE")
        print(f"  Epistemic (σ²_e): {sigma_e:.6f}  ← 模型分歧")

# ============================================
# 7. 不同频率的 σ²_a / σ²_e 对比曲线
# ============================================
freq_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
sigma_a_list = []
sigma_e_list = []

for freq in freq_list:
    t = np.linspace(0, 1, 100)
    batch_data = []
    for _ in range(10):
        phase = np.random.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * freq * t + phase)
        x += np.random.normal(0, 0.05, x.shape)
        batch_data.append(x)
    data_batch = np.stack(batch_data)[..., np.newaxis]
    data_batch_t = torch.tensor(data_batch, dtype=torch.float32)

    x_hats = []
    for k in range(K):
        x_hat = model(data_batch_t)
        x_hats.append(x_hat)
    x_hats = torch.stack(x_hats, dim=0)
    x_bar = x_hats.mean(dim=0)

    sigma_a = ((data_batch_t - x_hats) ** 2).mean().item()
    sigma_e = ((x_hats - x_bar) ** 2).mean().item()
    sigma_a_list.append(sigma_a)
    sigma_e_list.append(sigma_e)

# 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(freq_list, sigma_a_list, marker='s', color='steelblue', linewidth=2, label=r'$\sigma^2_a$ (Aleatoric)')
ax1.axvspan(0.5, 2.0, alpha=0.12, color='green')
ax1.axvspan(2.0, 8.0, alpha=0.12, color='red')
ax1.set_ylabel(r'$\sigma^2_a$ (RE mean)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Uncertainty Decomposition vs Signal Frequency')

ax2.plot(freq_list, sigma_e_list, marker='o', color='darkorange', linewidth=2, label=r'$\sigma^2_e$ (Epistemic)')
ax2.axvspan(0.5, 2.0, alpha=0.12, color='green', label='Training Range')
ax2.axvspan(2.0, 8.0, alpha=0.12, color='red', label='Unseen Range')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel(r'$\sigma^2_e$ (model disagreement)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
