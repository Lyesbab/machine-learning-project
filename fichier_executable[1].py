# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
df = pd.read_csv("ObesityDataSet.csv")
X = df.drop("NObeyesdad", axis=1)

y_pred = model.predict(X.head())# ici j'ai decide de tester les 5 premiers valeurs pour predire plus :
# veuiller mettre le nombre de valeurs que vous voulez predire par exemple pour 20 premiers valeurs
# changer a : y_pred = model.predict(X.head(20))

le = joblib.load("label_encoder.pkl")
decoded = le.inverse_transform(y_pred)

print(decoded)

# %%
