from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Cargar dataset
data = load_breast_cancer()

# Convertir a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = maligno, 1 = benigno

# Vista inicial
print(f"Shape: {df.shape}")
print(f"\nClases:\n{df['target'].value_counts()}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nInfo general:\n{df.info()}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")

# Guardar como CSV
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/breast_cancer.csv", index=False)
print("\nDataset guardado en data/raw/breast_cancer.csv")
