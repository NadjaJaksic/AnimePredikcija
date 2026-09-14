# Anime Predikcija

Projekat iz oblasti **mašinskog učenja i obrade podataka** čiji je cilj predikcija ocene i kvaliteta anime naslova na osnovu podataka iz **MyAnimeList** baze.

Projekat obuhvata pripremu i obradu podataka, treniranje neuronskih mreža i evaluaciju dobijenih modela. Korišćena su dva različita pristupa kako bi se ispitala mogućnost predikcije na osnovu strukturiranih podataka i tekstualnog sadržaja.

## Modeli

### MLP neuronske mreže

Za predikciju na osnovu numeričkih i kategorijalnih karakteristika animea korišćene su **MLP (Multi-Layer Perceptron) neuronske mreže**.

Korišćene su karakteristike poput broja epizoda, popularnosti, broja članova, broja favorita, ranga, izvora i starosne kategorije.

Implementirani su:

* regresioni model za predikciju numeričke ocene animea,
* klasifikacioni model za određivanje kategorije kvaliteta,
* multi-task model koji istovremeno rešava regresioni i klasifikacioni problem.

### NLP – Word2Vec + Bidirectional LSTM

Drugi deo projekta koristi **tekst sinopsisa animea** za određivanje njegove kategorije kvaliteta.

Tekst prolazi kroz preprocessing koji uključuje tokenizaciju, uklanjanje stop reči i lematizaciju. Za predstavljanje reči korišćen je **Word2Vec**, dok je za klasifikaciju sekvenci korišćena **Bidirectional LSTM neuronska mreža**.

Anime naslovi klasifikuju se u četiri kategorije:

**Bad | Average | Good | Excellent**

## Tehnologije

* **Python**
* **NumPy & Pandas** – obrada i analiza podataka
* **TensorFlow / Keras** – neuronske mreže
* **Scikit-learn** – preprocessing i evaluacija modela
* **NLTK** – obrada prirodnog jezika
* **Gensim / Word2Vec** – vektorska reprezentacija reči
* **Matplotlib** – vizualizacija rezultata

## Evaluacija modela

Za regresione modele korišćene su metrike **MAE, RMSE i R²**, dok su klasifikacioni modeli evaluirani pomoću **Accuracy, F1-score, Recall** i matrice konfuzije.

## Autori

**Nađa Jakšić**

**Aleksandra Golić**

Fakultet tehničkih nauka, Univerzitet u Novom Sadu
Informacioni inženjering

Projekat implementira dva modela za predikciju kvaliteta anime serija:
- **Model 1** — Višeslojna neuronska mreža 
- **Model 2** — Rekurentna neuronska mreža 

## Requirements

Sve biblioteke potrebne za pokretanje se nalaze u fajlu requirements.txt
```bash
pip install -r requirements.txt
```

## Pokretanje

1. Preuzeti dataset `anime.csv` i smestiti ga u isti folder kao `prediction.py`

link: https://www.kaggle.com/datasets/sazzadsiddiquelikhon/myanimelist-anime-database-july-2025/data 
 
2. Pokrenuti skriptu u cmd-u:

```bash
python prediction.py
```

