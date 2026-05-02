import numpy as np

# --- TEK NÖRON ---
girdi = np.array([1.5, 0.8, 2.1])    # 3 giriş özelliği
agirlik = np.array([0.4, -0.2, 0.7]) # her özelliğin ağırlığı
bias = 0.1

sonuc = np.dot(girdi, agirlik) + bias
print("Nöron çıktısı:", sonuc)

# Aktivasyon fonksiyonu (ReLU)
def relu(x):
    return max(0, x)

aktivasyon = relu(sonuc)
print("Aktivasyon sonrası:", aktivasyon)

# --- KATMAN (4 nöron aynı anda) ---
katman_agirliklari = np.array([
    [0.4, -0.2,  0.7],  # 1. nöron
    [0.1,  0.5, -0.3],  # 2. nöron
    [-0.6, 0.8,  0.2],  # 3. nöron
    [0.9, -0.1,  0.4],  # 4. nöron
])

katman_bias = np.array([0.1, -0.2, 0.3, 0.0])

katman_sonuc = np.dot(katman_agirliklari, girdi) + katman_bias
print("Katman çıktısı:", katman_sonuc)

katman_aktivasyon = np.maximum(0, katman_sonuc)
print("Katman aktivasyon:", katman_aktivasyon)