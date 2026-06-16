# 🎬 MovieLens Recommendation System using PySpark ALS
Sistem rekomendasi film berskala besar menggunakan algoritma **Alternating Least Squares (ALS)** berbasis **Matrix Factorization** di atas framework **Apache Spark (PySpark)**.
Proyek ini mencakup seluruh pipeline data mulai dari:
* Data Quality Checking
* Sparsity Analysis
* Data Preprocessing
* Model Training dengan ALS
* Evaluasi menggunakan metrik prediksi dan ranking
---

## Fitur Utama
### Skalabilitas Tinggi
Memanfaatkan **PySpark DataFrame** untuk memproses jutaan data interaksi pengguna dari dataset MovieLens.
### Analisis Sparsity
Menghitung tingkat kerapatan (*density*) dan kekosongan (*sparsity*) matriks user-item sebelum dan sesudah preprocessing.
### Data Filtering Criteria
Dataset difilter menggunakan kriteria berikut:
* User minimal memberikan **20 rating**
* Film minimal menerima **50 rating**
* Hanya menggunakan **positive ratings** (`rating ≥ 4`)

### Stratified Hold-Out Split
Pembagian data dilakukan secara proporsional untuk setiap user menggunakan **Spark Window Functions**.
* 80% data → Training
* 20% data positif terbaru → Testing
Pendekatan ini membantu mengurangi risiko **data leakage** pada proses evaluasi.

### Evaluasi Komprehensif
Model dievaluasi menggunakan:

#### Prediction Metrics
* RMSE (Root Mean Squared Error)
#### Ranking Metrics
* Hit Rate@20
* Precision@20
* Recall@20
---

## Dataset
Proyek menggunakan dataset **MovieLens Latest** (33M ratings atau versi besar lainnya). 
Link Dataset : https://www.kaggle.com/datasets/nabilaryamuzakki/movielens-latest-2026/versions/1

### `ratings.csv`
Berisi interaksi pengguna terhadap film.
| Kolom     | Deskripsi              |
| --------- | ---------------------- |
| userId    | ID pengguna            |
| movieId   | ID film                |
| rating    | Nilai rating           |
| timestamp | Waktu pemberian rating |

### `movies.csv`
Berisi metadata film.
| Kolom   | Deskripsi  |
| ------- | ---------- |
| movieId | ID film    |
| title   | Judul film |
| genres  | Genre film |
---

## Pipeline Implementation
### 1. Data Quality Checking
Tahap awal untuk memastikan kualitas data:
* Memeriksa schema dataset
* Mendeteksi missing values
* Menghitung jumlah user dan movie unik
* Memvalidasi integritas data
---

### 2. Exploratory Data Analysis (EDA)
Analisis distribusi:
* Aktivitas pengguna
* Popularitas film
* Distribusi rating
Visualisasi dilakukan menggunakan **Matplotlib** dan **Seaborn**.
---

### 3. Data Preprocessing

#### Filtering
* User ≥ 20 rating
* Movie ≥ 50 rating
* Rating ≥ 4
#### Sampling
Mengambil hingga **10.000 eligible users** untuk mengontrol penggunaan memori dan waktu komputasi.

#### Train-Test Split
Menggunakan pendekatan stratified split berbasis:
```python
Window.partitionBy("userId")
```
untuk menjaga proporsi data tiap pengguna.
---

### 4. ALS Model Training
Konfigurasi model:
```python
from pyspark.ml.recommendation import ALS
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=50,
    maxIter=10,
    regParam=0.05,
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)
```
---

### 5. Model Evaluation
#### Regression Evaluation
Menggunakan:
```text
RMSE (Root Mean Squared Error)
```

#### Top-N Recommendation Evaluation
Menghasilkan:
```python
recommendForAllUsers(20)
```
Kemudian dibandingkan dengan data aktual pada test set untuk menghitung:
* HitRate@20
* Precision@20
* Recall@20
---

## 📊 Sparsity Analysis
### Sebelum Preprocessing
| Kondisi Dataset | Total Sel Matriks | Total Rating | Density | Sparsity |
| --------------- | ----------------- | ------------ | ------- | -------- |
| Raw Dataset     | 27,550,028,025    | 33,832,162   | 0.1228% | 99.8772% |
### Setelah Filtering
| Kondisi Dataset     | Total Sel Matriks | Total Rating | Density | Sparsity |
| ------------------- | ----------------- | ------------ | ------- | -------- |
| Post-Filter Dataset | 160,560,000       | 1,612,868    | 1.0045% | 98.9955% |
---

## 📈 Modeling Results
```text
============================================================
         RANGKUMAN METRIC & DIMENSI MODELING ALS
============================================================

SAMPEL DATASET:
  • Total Users Terlibat     : 10,000
  • Total Movies Terlibat    : 16,056
  • Jumlah Rating (Train)    : 1,612,868
  • Jumlah Rating (Test)     : 183,155
------------------------------------------------------------
METRIK EVALUASI:
  • RMSE                     : 0.6785
  • HitRate@20               : 0.1937 (19.37%)
  • Precision@20             : 0.0127 (1.27%)
  • Recall@20                : 0.0232 (2.32%)
============================================================
```



