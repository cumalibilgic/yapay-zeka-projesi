import numpy as np

# Sinir ağında her şey sayı dizisi (array)
girdi = np.array([1.5, 0.8, 2.1])    # 3 giriş özelliği
agirlik = np.array([0.4, -0.2, 0.7]) # her özelliğin ağırlığı
bias = 0.1

# Nöronun yaptığı tek şey bu:
sonuc = np.dot(girdi, agirlik) + bias
print("Nöron çıktısı:", sonuc)