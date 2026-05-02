import numpy as np

veriler = [
    ([2.0,  1.5,  3.0], 1),
    ([-1.0, -2.0, -0.5], 0),
    ([0.5,  1.0,  0.8], 1),
    ([-0.3, -1.5, -2.0], 0),
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)
agirlik = np.random.randn(3)
bias = 0.0

print("Başlangıç ağırlıkları:", agirlik)

ogrenme_hizi = 0.1

for epoch in range(100):
    toplam_kayip = 0
    for girdi, dogru_cevap in veriler:
        girdi = np.array(girdi)
        tahmin = sigmoid(np.dot(agirlik, girdi) + bias)
        kayip = (tahmin - dogru_cevap) ** 2
        toplam_kayip += kayip
        hata = tahmin - dogru_cevap
        agirlik -= ogrenme_hizi * hata * girdi
        bias -= ogrenme_hizi * hata
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} — Kayıp: {toplam_kayip:.4f}")

print("\nSon ağırlıklar:", agirlik)
print("Son bias:", bias)

# --- TEST ---
print("\n--- TEST ---")

test_verileri = [
    ([1.0,  2.0,  1.5], 1),
    ([-2.0, -1.0, -3.0], 0),
    ([0.2,  0.4,  0.6], 1),
    ([-0.5, -0.8, -0.3], 0),
]

dogru = 0

for girdi, dogru_cevap in test_verileri:
    girdi = np.array(girdi)
    tahmin = sigmoid(np.dot(agirlik, girdi) + bias)
    sonuc = 1 if tahmin >= 0.5 else 0
    durum = "✓" if sonuc == dogru_cevap else "✗"
    print(f"{durum} Girdi: {girdi}  →  Tahmin: {tahmin:.2f}  →  {'POZİTİF' if sonuc == 1 else 'NEGATİF'}")
    if sonuc == dogru_cevap:
        dogru += 1

print(f"\nDoğruluk: {dogru}/{len(test_verileri)}")