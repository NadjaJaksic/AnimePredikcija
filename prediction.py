# Nađa Jakšić IN33/2023
# %% Biblioteke

import numpy as np
import pandas as pd

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                              r2_score, accuracy_score, f1_score,
                              recall_score, confusion_matrix)

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

import tensorflow as tf

import matplotlib.pyplot as plt

import random, os

# %% Generator vrednosti

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

gpsv = np.random.default_rng(42)

# %% Učitavanje i uređivanje podataka

print("=" * 60)
print("UCITAVANJE I UREĐIVANJE PODATAKA")
print("=" * 60)

podaci1 = pd.read_csv("anime.csv")

print("\nBroj kolona sa nedostajućim vrednostima:", podaci1.isnull().any().sum())
print("-" * 60)

podaci1 = podaci1.dropna(subset=["score"])
print("\nZadrzavanje samo redova sa ocenom (score)...")

# Popunjavanje nedostajućih vrednosti medijanom
for k in ["episodes", "rank", "aired_prop_from_year", "scored_by"]:
    podaci1[k] = podaci1[k].fillna(podaci1[k].median())

print("Popunjavanje nedostajućih vrednosti medijanom za kolone: episodes, rank, aired_prop_from_year, scored_by...")

# One-hot enkodiranje kategorijskih kolona
podaci1 = pd.get_dummies(podaci1, columns=["source", "rating"], dummy_na=True)

print("One-hot enkodiranje kolona: source, rating...\n")
print("-" * 60)

# Kreiranje klasa kvaliteta na osnovu ocene
bins   = [0, 5, 6.5, 7.5, 10]
labele = ["Bad", "Average", "Good", "Excellent"]
podaci1["klasa"] = pd.cut(podaci1["score"], bins=bins, labels=labele)
podaci1 = podaci1.dropna(subset=["klasa"])

print("\n=== RASPODELA KLASA ===")
print(podaci1["klasa"].value_counts().sort_index())


klasa_mapa     = {"Bad": 0, "Average": 1, "Good": 2, "Excellent": 3}
klasa_mapa_obr = {0: "Bad", 1: "Average", 2: "Good", 3: "Excellent"}
podaci1["klasa_oznaka"] = podaci1["klasa"].map(klasa_mapa)

# Vizualizacija raspodele klasa
ocene = podaci1["score"].dropna()

plt.figure(figsize=(8, 5))
plt.hist(ocene, bins=40, color="#7B5EA7", edgecolor="white")
plt.title("Srednja vrednost ocena")
plt.xlabel("Ocena")
plt.ylabel("Broj anime-a")
plt.axvline(ocene.mean(),   color="red",    linestyle="--", label=f"Prosek: {ocene.mean():.2f}")
plt.axvline(ocene.median(), color="orange", linestyle="--", label=f"Medijana: {ocene.median():.2f}")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
boje = ["#E07A5F", "#F2CC8F", "#81B29A", "#7B5EA7"]
raspodela = podaci1["klasa"].value_counts().sort_index()
plt.bar(list(klasa_mapa.keys()), [raspodela[v] for v in klasa_mapa.keys()],
        color=boje, edgecolor="white")
plt.title("Raspodela anime-a po klasama kvaliteta")
plt.xlabel("Klasa")
plt.ylabel("Broj anime-a")
plt.tight_layout()
plt.show()


print("=" * 60)
print("PRIPREMA PODATAKA ZA OBUČAVANJE")
print("=" * 60)

bazne_kolone = ["episodes", "members", "popularity", "rank",
                "favorites", "scored_by", "aired_prop_from_year"]
ohe_kolone   = [k for k in podaci1.columns
                if k.startswith("source_") or k.startswith("rating_")]
ulazne_kolone = bazne_kolone + ohe_kolone

ob_ulaz_kard         = len(ulazne_kolone)
ob_cilj_jedinst_kard = podaci1["klasa_oznaka"].nunique()

skup_obucavanje1 = podaci1.sample(frac=0.7, random_state=gpsv.bit_generator)
ostalo           = podaci1.drop(index=skup_obucavanje1.index)
skup_val1        = ostalo.sample(frac=0.5, random_state=42)
skup_test1       = ostalo.drop(index=skup_val1.index)


print("\n=== KORIŠĆENE ULAZNE KOLONE ===")
print("\nBroj ulaznih numeričkih kolona:", len(bazne_kolone))
print("Broj ulaznih kategorijskih kolona:", len(ohe_kolone))
print("Ukupan broj ulaznih kolona:", ob_ulaz_kard)
print("\n" + "-" * 60)


# Normalizacija 
X_ob1 = skup_obucavanje1[ulazne_kolone].astype(float)
X_te1 = skup_test1[ulazne_kolone].astype(float)
X_val1   = skup_val1[ulazne_kolone].astype(float)
min_v  = X_ob1.min()
max_v  = X_ob1.max()
raspon = (max_v - min_v)
raspon[raspon == 0] = 1

ulaz_ob1 = ((X_ob1 - min_v) / raspon).values
ulaz_te1 = ((X_te1 - min_v) / raspon).values
ulaz_val1 = ((X_val1 - min_v) / raspon).values


ocena_ob  = (skup_obucavanje1["score"].values / 10.0).reshape(-1, 1)
klasa_ob  = pd.get_dummies(skup_obucavanje1["klasa_oznaka"]).reindex(
                columns=[0,1,2,3], fill_value=0).values
ocena_val = (skup_val1["score"].values / 10.0).reshape(-1, 1)
klasa_val = pd.get_dummies(skup_val1["klasa_oznaka"]).reindex(
                columns=[0,1,2,3], fill_value=0).values

print("Oblik skupa za obučavanje (Model 1):", ulaz_ob1.shape)
print("Oblik validacionog skupa (Model 1):", ulaz_val1.shape)
print("Oblik skupa za testiranje (Model 1):", ulaz_te1.shape)
print()



print("=" * 60)
print("FORMIRANJE MULTI-TASK NEURONSKE MREŽE")
print("=" * 60)

# Ulazni sloj
ulaz_sloj = Input(shape=(ob_ulaz_kard,))

# Zajednički skriveni slojevi

x = Dense(128, activation=tf.nn.leaky_relu)(ulaz_sloj)
x = Dense(128, activation=tf.nn.leaky_relu)(x)
x = Dense(64, activation=tf.nn.leaky_relu)(x)

# Izlazni sloj za regresiju (ocena)
izlaz_reg  = Dense(1, activation="sigmoid", name="regresija")(x)

# Izlazni sloj za klasifikaciju (klasa kvaliteta)
izlaz_klas = Dense(ob_cilj_jedinst_kard, activation="softmax", name="klasifikacija")(x)

nm_mt = Model(inputs=ulaz_sloj, outputs=[izlaz_reg, izlaz_klas])
nm_mt.compile(
    optimizer=Adam(),
    loss={"regresija": "mse", "klasifikacija": "categorical_crossentropy"},
    metrics={"regresija": "mae", "klasifikacija": "categorical_accuracy"}
)

nm_mt.summary()
print()


# %% Obučavanje multi-task modela

print("=" * 60)
print("OBUČAVANJE MULTI-TASK MODELA")
print("=" * 60)
istorija = nm_mt.fit(
    ulaz_ob1,
    {"regresija": ocena_ob, "klasifikacija": klasa_ob},
    epochs=100, batch_size=64, verbose=1,
    validation_data=(ulaz_val1, {"regresija": ocena_val, "klasifikacija": klasa_val})
)
print()

# %%Vizualizacija gubitka tokom treniranja

plt.figure(figsize=(8, 5))
plt.plot(istorija.history["loss"],     label="Trening gubitak",     color="#7B5EA7")
plt.plot(istorija.history["val_loss"], label="Validacioni gubitak", color="#E07A5F")
plt.xlabel("Epoha")
plt.ylabel("Loss")
plt.title("Loss tokom treniranja — Multi-task model")
plt.legend()
plt.tight_layout()
plt.show()

# %% Predikcija i evaluacija multi-task modela

print("=" * 60)
print("PREDIKCIJA I EVALUACIJA MULTI-TASK MODELA")
print("=" * 60)

# Predikcija 
pred_ocena_norm, pred_klas_prob = nm_mt.predict(ulaz_te1, verbose=0)
pred_ocena      = (pred_ocena_norm * 10.0).ravel()
stvarna_ocena   = skup_test1["score"].values
pred_klas       = np.argmax(pred_klas_prob, axis=1)
stvarna_klas    = skup_test1["klasa_oznaka"].values


# Evaluacija
mae_mt  = mean_absolute_error(stvarna_ocena, pred_ocena)
rmse_mt = root_mean_squared_error(stvarna_ocena, pred_ocena)
r2_mt   = r2_score(stvarna_ocena, pred_ocena)
acc_mt  = accuracy_score(stvarna_klas, pred_klas)
f1_mt   = f1_score(stvarna_klas, pred_klas, average="weighted")
rec_mt  = recall_score(stvarna_klas, pred_klas, average="weighted")
cm_mt   = confusion_matrix(stvarna_klas, pred_klas)

print("\nRegresione metrike:")
print(f"  MAE  = {mae_mt:.4f}")
print(f"  RMSE = {rmse_mt:.4f}")
print(f"  R²   = {r2_mt:.4f}")
print("-" * 60)
print("\nKlasifikacione metrike:")
print(f"  Accuracy = {acc_mt:.4f}")
print(f"  F1-score = {f1_mt:.4f}")
print(f"  Recall   = {rec_mt:.4f}")
print("-" * 60)
print("\nMatrica konfuzije (multi-task):")
print("  Klase:", list(klasa_mapa.keys()))
print(cm_mt)
print()

# %% Vizualizacija rezultata multi-task modela

plt.figure(figsize=(7, 6))
plt.scatter(stvarna_ocena, pred_ocena, alpha=0.3, s=10, color="#7B5EA7")
plt.plot([1, 10], [1, 10], color="#E07A5F", linestyle="--", label="Idealna predikcija")
plt.xlabel("Stvarna ocena")
plt.ylabel("Predviđena ocena")
plt.title("Stvarne vs. predviđene ocene — Multi-task model")
plt.legend()
plt.tight_layout()
plt.show()

# Vizualizacija matrice konfuzije

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_mt, interpolation="nearest", cmap="BuPu")
plt.colorbar(im, ax=ax)
klase_nazivi = list(klasa_mapa.keys())
ax.set_xticks(range(4)); ax.set_xticklabels(klase_nazivi)
ax.set_yticks(range(4)); ax.set_yticklabels(klase_nazivi)
for i in range(cm_mt.shape[0]):
    for j in range(cm_mt.shape[1]):
        boja = "white" if cm_mt[i, j] > cm_mt.max() / 2 else "black"
        ax.text(j, i, str(cm_mt[i, j]), ha="center", va="center",
                color=boja, fontsize=12, fontweight="bold")
ax.set_xlabel("Predviđena klasa")
ax.set_ylabel("Stvarna klasa")
ax.set_title("Matrica konfuzije — Multi-task model")
plt.tight_layout()
plt.show()


# %% Zasebne specijalizovane mreže - regresiona mreža

print("=" * 60)
print("ZASEBNE SPECIJALIZOVANE MREŽE - REGRESIONA")
print("=" * 60)

#--------------------------------
# Zasebna regresiona mreža
#--------------------------------

ulaz_r = Input(shape=(ob_ulaz_kard,))
xr = Dense(128, activation=tf.nn.leaky_relu)(ulaz_r)
xr = Dense(128, activation=tf.nn.leaky_relu)(xr)
xr = Dense(64, activation=tf.nn.leaky_relu)(xr)
nm_reg = Model(inputs=ulaz_r, outputs=Dense(1, activation="sigmoid")(xr))
nm_reg.compile(optimizer=Adam(), 
               loss="mse", 
               metrics=["mae"])

print("\nObučavanje zasebne regresione mreže...")
print("-" * 60)

istorija_r = nm_reg.fit(
    ulaz_ob1, ocena_ob,
    epochs=100, batch_size=64, verbose=1,
    validation_data=(ulaz_val1, ocena_val)
)
pred_r     = nm_reg.predict(ulaz_te1, verbose=0).ravel() * 10.0
mae_r      = mean_absolute_error(stvarna_ocena, pred_r)
rmse_r     = root_mean_squared_error(stvarna_ocena, pred_r)
r2_r       = r2_score(stvarna_ocena, pred_r)

# %% Evaluacija zasebne regresione mreže

print("\nRegresione metrike (zasebna mreža):")
print(f"  MAE  = {mae_r:.4f}  |  RMSE = {rmse_r:.4f}  |  R² = {r2_r:.4f}")

# %% Vizualizacija gubitka tokom treniranja zasebne regresione mreže


plt.figure(figsize=(8, 4))
plt.plot(istorija_r.history["loss"],     label="Trening gubitak",     color="#7B5EA7")
plt.plot(istorija_r.history["val_loss"], label="Validacioni gubitak", color="#E07A5F")
plt.xlabel("Epoha")
plt.ylabel("Loss (MSE)")
plt.title("Loss tokom treniranja — Zasebna regresiona mreža")
plt.legend()
plt.tight_layout()
plt.show()



# %% Vizualizacija rezultata zasebne regresione mreže

plt.figure(figsize=(7, 6))
plt.scatter(stvarna_ocena, pred_r, alpha=0.3, s=10, color="#3AAFA9")
plt.plot([1, 10], [1, 10], color="#E07A5F", linestyle="--", label="Idealna predikcija")
plt.xlabel("Stvarna ocena")
plt.ylabel("Predviđena ocena")
plt.title("Stvarne vs. predviđene ocene — Zasebna regresiona mreža")
plt.legend()
plt.tight_layout()
plt.show()

# %% Zasebne specijalizovane mreže - klasifikaciona mreža

print("=" * 60)
print("ZASEBNE SPECIJALIZOVANE MREŽE - KLASIFIKACIONA")
print("=" * 60)

# Zasebna klasifikaciona mreža

ulaz_k = Input(shape=(ob_ulaz_kard,))
xk = Dense(128, activation=tf.nn.leaky_relu)(ulaz_k)
xk = Dense(128, activation=tf.nn.leaky_relu)(xk)
xk = Dense(64, activation=tf.nn.leaky_relu)(xk)
nm_klas = Model(inputs=ulaz_k, outputs=Dense(ob_cilj_jedinst_kard, activation="softmax")(xk))
nm_klas.compile(optimizer=Adam(),
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"])

print("\nObučavanje zasebne klasifikacione mreže...")
print("-" * 60)

istorija_k = nm_klas.fit(
    ulaz_ob1, klasa_ob,
    epochs=100, batch_size=64, verbose=1,
    validation_data=(ulaz_val1, klasa_val)
)
pred_k     = np.argmax(nm_klas.predict(ulaz_te1, verbose=0), axis=1)
acc_k      = accuracy_score(stvarna_klas, pred_k)
f1_k       = f1_score(stvarna_klas, pred_k, average="weighted")
rec_k      = recall_score(stvarna_klas, pred_k, average="weighted")
cm_k       = confusion_matrix(stvarna_klas, pred_k)

# %% Evaluacija zasebne klasifikacione mreže

print("\nKlasifikacione metrike (zasebna mreža):")
print(f"  Accuracy = {acc_k:.4f}  |  F1 = {f1_k:.4f}  |  Recall = {rec_k:.4f}")
print("\nMatrica konfuzije (zasebna mreža):")
print(cm_k)
print()

# %% Vizualizacija gubitka tokom treniranja zasebne klasifikacione mreže

plt.figure(figsize=(8, 4))
plt.plot(istorija_k.history["loss"],     label="Trening gubitak",     color="#7B5EA7")
plt.plot(istorija_k.history["val_loss"], label="Validacioni gubitak", color="#E07A5F")
plt.xlabel("Epoha")
plt.ylabel("Loss (Categorical Crossentropy)")
plt.title("Loss tokom treniranja — Zasebna klasifikaciona mreža")
plt.legend()
plt.tight_layout()
plt.show()

# %% Vizualizacija rezultata zasebne klasifikacione mreže

# Vizualizacija matrice konfuzije 

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_k, interpolation="nearest", cmap="BuPu")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(4)); ax.set_xticklabels(klase_nazivi)
ax.set_yticks(range(4)); ax.set_yticklabels(klase_nazivi)
for i in range(cm_k.shape[0]):
    for j in range(cm_k.shape[1]):
        boja = "white" if cm_k[i, j] > cm_k.max() / 2 else "black"
        ax.text(j, i, str(cm_k[i, j]), ha="center", va="center",
                color=boja, fontsize=12, fontweight="bold")
ax.set_xlabel("Predviđena klasa")
ax.set_ylabel("Stvarna klasa")
ax.set_title("Matrica konfuzije — Zasebna klasifikaciona mreža")
plt.tight_layout()
plt.show()

# %% Tabela poređenja

print("=" * 60)
print("POREĐENJE: MULTI-TASK vs. ZASEBNI MODELI")
print("=" * 60)
print(f"\n{'Metrika':<30} {'Multi-task':>12} {'Zasebni':>12}")
print("-" * 56)
print(f"{'Regresija — MAE':<30} {mae_mt:>12.4f} {mae_r:>12.4f}")
print(f"{'Regresija — RMSE':<30} {rmse_mt:>12.4f} {rmse_r:>12.4f}")
print(f"{'Regresija — R²':<30} {r2_mt:>12.4f} {r2_r:>12.4f}")
print(f"{'Klasifikacija — Accuracy':<30} {acc_mt:>12.4f} {acc_k:>12.4f}")
print(f"{'Klasifikacija — F1-score':<30} {f1_mt:>12.4f} {f1_k:>12.4f}")
print(f"{'Klasifikacija — Recall':<30} {rec_mt:>12.4f} {rec_k:>12.4f}")
print()

#Vizualizacija poređenja

sirina = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

met_reg  = ["MAE", "RMSE", "R²"]
vr_mt_r  = [mae_mt, rmse_mt, r2_mt]
vr_z_r   = [mae_r,  rmse_r,  r2_r]
x1 = np.arange(len(met_reg))
ax1.bar(x1 - sirina/2, vr_mt_r, sirina, label="Multi-task",   color="#7B5EA7", edgecolor="white")
ax1.bar(x1 + sirina/2, vr_z_r,  sirina, label="Zasebna reg.", color="#3AAFA9", edgecolor="white")
ax1.set_xticks(x1); ax1.set_xticklabels(met_reg)
ax1.set_title("Regresione metrike")
ax1.legend()

met_klas = ["Accuracy", "F1-score", "Recall"]
vr_mt_k  = [acc_mt, f1_mt, rec_mt]
vr_z_k   = [acc_k,  f1_k,  rec_k]
x2 = np.arange(len(met_klas))
ax2.bar(x2 - sirina/2, vr_mt_k, sirina, label="Multi-task",    color="#7B5EA7", edgecolor="white")
ax2.bar(x2 + sirina/2, vr_z_k,  sirina, label="Zasebna klas.", color="#E07A5F", edgecolor="white")
ax2.set_xticks(x2); ax2.set_xticklabels(met_klas)
ax2.set_title("Klasifikacione metrike")
ax2.legend()

plt.suptitle("Poređenje: Multi-task vs. Zasebni modeli")
plt.tight_layout()
plt.show()


# MODEL 2 — Rekurentna neuronska mreža (LSTM)


# %% Biblioteke i korpusi
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Masking, Dropout,Bidirectional
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import class_weight

import warnings
warnings.filterwarnings("ignore")


# %% Učitavanje i sređivanje podataka

anime = pd.read_csv("anime.csv")

print("Skup podataka (naslov,sinopsis,i ocena):")
print(anime[["title", "synopsis", "score"]].head())

print("\nKolone i tipovi (sinopsis i ocena):")
print(anime[["synopsis", "score"]].dtypes)

print("\nNedostajuce vrednosti:")
print(anime[["synopsis", "score"]].isna().sum())


# %% Analiza podataka

ocene = anime["score"].dropna()

anime_sa_syn = anime.loc[anime["synopsis"].notna() & (anime["synopsis"] != "")]
anime_sa_syn["duzina_sinopsisa"] = anime_sa_syn["synopsis"].str.split().str.len()

plt.figure(figsize=(8, 5))
plt.hist(anime_sa_syn["duzina_sinopsisa"], bins=50, color="#3AAFA9", edgecolor="white")
plt.title("Duzina sinopsisa")
plt.xlabel("Broj reci")
plt.ylabel("Broj anime-a")
plt.axvline(anime_sa_syn["duzina_sinopsisa"].median(), color="red",
            linestyle="--", label=f"Medijana: {anime_sa_syn['duzina_sinopsisa'].median():.0f}")
plt.legend()
plt.tight_layout()
plt.show()

print("\nStatistike ocene")
print(ocene.describe().round(3))

print("\nStatistike duzine sinopsisa")
print(anime_sa_syn["duzina_sinopsisa"].describe().round(1))


# %% Odabir podataka

anime_odabir = anime.loc[
    anime["synopsis"].notna() &
    (anime["synopsis"] != "") &
    anime["score"].notna()
].reset_index(drop=True)

print("\nBroj zapisa za analizu:", len(anime_odabir))

klase = {0: "Bad", 1: "Average", 2: "Good", 3: "Excellent"}

def mapiraj_kvalitet(ocena):
    if ocena < 5.5:
        return 0
    elif ocena < 6.5:
        return 1
    elif ocena < 7.5:
        return 2
    else:
        return 3

anime_odabir["kvalitet"] = anime_odabir["score"].apply(mapiraj_kvalitet).astype(int)
# %% Podaci - 70% trening, 15% validacija, 15% test

skup_trening    = anime_odabir.sample(frac=0.70, random_state=42)
skup_ostatak    = anime_odabir.drop(index=skup_trening.index)
skup_validacija = skup_ostatak.sample(frac=0.50, random_state=42)
skup_test       = skup_ostatak.drop(index=skup_validacija.index)

print("\nTrening:   ", len(skup_trening), "uzoraka")
print("Validacija:", len(skup_validacija), "uzoraka")
print("Test:      ", len(skup_test), "uzoraka")


# %% Priprema tokena

stop_reci = stopwords.words("english")
lematizator = WordNetLemmatizer()
recenice = skup_trening["synopsis"].tolist()

for i in range(len(recenice)):
    sinopsis = recenice[i]
    recenice[i] = [lematizator.lemmatize(token).lower()
                   for token in word_tokenize(sinopsis.replace("\r\n", " ").replace("\n", " "))
                   if token.lower() not in stop_reci and token.isalpha()]

print("Primeri tokena:")
print(recenice[0:4])


# %% Word2Vec trening

model_w2v = Word2Vec(recenice, vector_size=64, epochs=10, min_count=2, window=5, seed=42, workers=1)

rec = "fight"
print("\nNajslicnije reci za:", rec)
print(model_w2v.wv.most_similar(rec))


# %% Vektorizacija sinopsisa

VEKTOR_VELICINA = 64
duzine_tokena = [len(t) for t in recenice]
MAX_DUZINA = int(np.percentile(duzine_tokena, 95))

def sinopsis_u_sekvencu(tokeni, model_wv, max_duzina, vektor_velicina):
    sekvenca = np.zeros((max_duzina, vektor_velicina), dtype="float32")
    for j, token in enumerate(tokeni[:max_duzina]):
        if token in model_wv:
            sekvenca[j] = model_wv[token]
    return sekvenca

def vektorizuj(sinopsisi, model_wv, max_duzina, vektor_velicina):
    tokeni = []
    for sinopsis in sinopsisi:
        tokeni.append([lematizator.lemmatize(token).lower()
                       for token in word_tokenize(sinopsis.replace("\r\n", " ").replace("\n", " "))
                       if token.lower() not in stop_reci and token.isalpha()])
    return np.array([sinopsis_u_sekvencu(t, model_wv, max_duzina, vektor_velicina)
                     for t in tokeni], dtype="float32")

X_trening    = np.array([sinopsis_u_sekvencu(t, model_w2v.wv, MAX_DUZINA, VEKTOR_VELICINA)
                         for t in recenice], dtype="float32")
X_validacija = vektorizuj(skup_validacija["synopsis"].tolist(), model_w2v.wv, MAX_DUZINA, VEKTOR_VELICINA)
X_test       = vektorizuj(skup_test["synopsis"].tolist(),       model_w2v.wv, MAX_DUZINA, VEKTOR_VELICINA)

y_trening    = skup_trening["kvalitet"].values
y_validacija = skup_validacija["kvalitet"].values
y_test       = skup_test["kvalitet"].values

y_trening_matricno    = to_categorical(y_trening,    num_classes=4)
y_validacija_matricno = to_categorical(y_validacija, num_classes=4)
y_test_matricno       = to_categorical(y_test,       num_classes=4)

print("\nOblik X_trening:   ", X_trening.shape)
print("Oblik X_validacija:", X_validacija.shape)
print("Oblik X_test:      ", X_test.shape)


# %% Racunanje težina klasa

tezine_niz = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_trening),
    y=y_trening)
tezine_klasa = dict(enumerate(tezine_niz))

print("\nTezine klasa:")
for k, v in klase.items():
    print(" ", v, ":", round(tezine_klasa[k], 4))


# %% Treniranje mreže

broj_epoha = 20

rnm = Sequential()
rnm.add(Input(shape=(MAX_DUZINA, VEKTOR_VELICINA)))
rnm.add(Masking(mask_value=0.0))
rnm.add(Bidirectional(LSTM(64, return_sequences=True)))
rnm.add(Dropout(0.3))
rnm.add(Bidirectional(LSTM(32)))
rnm.add(Dropout(0.3))
rnm.add(Dense(16, activation="relu"))
rnm.add(Dense(4, activation="softmax"))
rnm.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
rnm.summary()

rnm_rezultati = rnm.fit(X_trening, y_trening_matricno,
                        epochs=broj_epoha,
                        batch_size=64,
                        validation_data=(X_validacija, y_validacija_matricno),
                        class_weight=tezine_klasa)


# %% Analiza performansi

plt.figure(figsize=(8, 6))
plt.plot(range(1, broj_epoha + 1), rnm_rezultati.history["loss"],     "o--", label="obucavanje", color="#7B5EA7")
plt.plot(range(1, broj_epoha + 1), rnm_rezultati.history["val_loss"], "o--", label="validacija",  color="#E07A5F")
plt.xlim(0)
plt.ylim(0)
plt.title("Performanse — Class Weight")
plt.xticks([i for i in range(1, broj_epoha + 1) if i % 5 == 0])
plt.xlabel("Epoha")
plt.ylabel("Gubitak")
plt.legend(title="Faza")
plt.tight_layout()
plt.show()


# %% Evaluacija

y_pred = np.argmax(rnm.predict(X_test), axis=1)

print("\nIzvestaj klasifikacije:")
print(classification_report(y_test, y_pred,
                            target_names=[klase[k] for k in sorted(klase)],
                            digits=4))

matrica_konfuzije = confusion_matrix(y_test, y_pred)
print("Matrica konfuzije:")
print(matrica_konfuzije)

klase_nazivi = [klase[k] for k in sorted(klase)]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrica_konfuzije, cmap="BuPu")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(klase_nazivi)
ax.set_yticklabels(klase_nazivi)
for i in range(matrica_konfuzije.shape[0]):
    for j in range(matrica_konfuzije.shape[1]):
        boja = "white" if matrica_konfuzije[i, j] > matrica_konfuzije.max() / 2 else "black"
        ax.text(j, i, str(matrica_konfuzije[i, j]), ha="center", va="center",
                color=boja, fontsize=12, fontweight="bold")
ax.set_xlabel("Predvidjena klasa")
ax.set_ylabel("Stvarna klasa")
ax.set_title("Matrica konfuzije — Class Weight")
plt.tight_layout()
plt.show()

f1_lstm_weighted = f1_score(y_test, y_pred, average="weighted")
f1_lstm_macro    = f1_score(y_test, y_pred, average="macro")

print("\nAccuracy:   ", round(accuracy_score(y_test, y_pred), 4))
print("F1 weighted:", round(f1_lstm_weighted, 4))
print("F1 macro:   ", round(f1_lstm_macro, 4))


# %% Poređenje F1 metrike — sve mreže

modeli  = ["Multi-task\n(MLP)", "Zasebna klas.\n(MLP)", "LSTM"]
f1_vred = [f1_mt, f1_k, f1_lstm_weighted]
boje_f1 = ["#7B5EA7", "#E07A5F", "#3AAFA9"]

plt.figure(figsize=(8, 5))
bars = plt.bar(modeli, f1_vred, color=boje_f1, edgecolor="white", width=0.5)
for bar, vred in zip(bars, f1_vred):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{vred:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.ylim(0, 1.05)
plt.ylabel("F1-score (weighted)")
plt.title("Poređenje F1 metrike — sve mreže")
plt.tight_layout()
plt.show()