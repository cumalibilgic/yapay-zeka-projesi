import numpy as np

# --- TEK NÖRON ---
girdi = np.array([1.5, 0.8, 2.1])
agirlik = np.array([0.4, -0.2, 0.7])
bias = 0.1

sonuc = np.dot(girdi, agirlik) + bias
print("Nöron çıktısı:", sonuc)

def relu(x):
    return max(0, x)

aktivasyon = relu(sonuc)
print("Aktivasyon sonrası:", aktivasyon)

# --- 1. KATMAN (4 nöron) ---
katman_agirliklari = np.array([
    [0.4, -0.2,  0.7],
    [0.1,  0.5, -0.3],
    [-0.6, 0.8,  0.2],
    [0.9, -0.1,  0.4],
])

katman_bias = np.array([0.1, -0.2, 0.3, 0.0])

katman_sonuc = np.dot(katman_agirliklari, girdi) + katman_bias
print("Katman çıktısı:", katman_sonuc)

katman_aktivasyon = np.maximum(0, katman_sonuc)
print("Katman aktivasyon:", katman_aktivasyon)

# --- 2. KATMAN (3 nöron) ---
katman2_agirliklari = np.array([
    [0.3, -0.5,  0.2,  0.8],
    [-0.4, 0.6,  0.9, -0.1],
    [0.7,  0.2, -0.3,  0.5],
])

katman2_bias = np.array([0.1, 0.0, -0.2])

katman2_sonuc = np.dot(katman2_agirliklari, katman_aktivasyon) + katman2_bias
print("2. Katman çıktısı:", katman2_sonuc)

katman2_aktivasyon = np.maximum(0, katman2_sonuc)
print("2. Katman aktivasyon:", katman2_aktivasyon)

# --- ÇIKTI KATMANI ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

cikti_agirliklari = np.array([0.5, -0.3, 0.8])
cikti_bias = 0.0

cikti_sonuc = np.dot(cikti_agirliklari, katman2_aktivasyon) + cikti_bias
tahmin = sigmoid(cikti_sonuc)
print("Tahmin (0-1 arası):", tahmin)

if tahmin >= 0.5:
    print("Sonuç: POZİTİF")
else:
    print("Sonuç: NEGATİF")