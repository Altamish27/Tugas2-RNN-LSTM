# Tugas 2: RNN & LSTM - Deep Learning

## Identitas Anggota
**Anggota Kelompok**
- Raffi Adzril Alfaiz (2308355)
- Hasbi Haqqul Fikri (2308355)
- Muhammad Bintang Eighista (2304137)
- Muhammad Isa Abdullah (2303508) 


Repository ini berisi implementasi **Recurrent Neural Networks (RNN)** dan **Long Short-Term Memory (LSTM)** untuk dua task utama: **Question-Answering Chatbot** dan **Sentiment Classification** menggunakan dataset Bahasa Indonesia.

---

## 📋 Daftar Isi

- [Overview](#overview)
- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Deskripsi Proyek](#deskripsi-proyek)
  - [1. RNN & LSTM Chatbot Q&A](#1-rnn--lstm-chatbot-qa)
  - [2. RNN & LSTM Sentiment Classification](#2-rnn--lstm-sentiment-classification)
- [Hasil dan Model](#hasil-dan-model)
- [Cara Menjalankan](#cara-menjalankan)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Kontributor](#kontributor)

---

## 🎯 Overview

Proyek ini mengimplementasikan dan membandingkan performa antara:
- **RNN (Simple RNN)** vs **LSTM** untuk task Question-Answering
- **RNN (Simple RNN)** vs **LSTM** vs **Bi-LSTM** untuk task Sentiment Classification

Semua model dilatih menggunakan dataset Bahasa Indonesia untuk mendemonstrasikan kemampuan sequence modeling dalam Natural Language Processing (NLP).

---

## 📁 Struktur Proyek

```
Tugas2-RNN-LSTM/
│
├── LSTM-CHATBOT/                          # LSTM untuk Q&A Chatbot
│   ├── LSTM_Chatbot.ipynb                 # Notebook implementasi LSTM Chatbot
│   ├── dataset/
│   │   ├── df_train.json                  # Dataset training IndoQA
│   │   └── df_val.json                    # Dataset validasi IndoQA
│   └── Output Model LSTM/
│       ├── best_lstm_model.h5             # Model terbaik (checkpoint)
│       └── final_lstm_model.h5            # Model final
│
├── RNN-CHATBOT-INDONESIA-CONTEXT-QUESTION-ANSWER/  # RNN untuk Q&A Chatbot
│   ├── IPYNB/
│   │   └── RNN_Chatbot_QA_Indonesian.ipynb  # Notebook implementasi RNN Chatbot
│   ├── dataset/
│   │   ├── df_train.json                  # Dataset training IndoQA
│   │   └── df_val.json                    # Dataset validasi IndoQA
│   └── Output Model RNN/
│       ├── best_rnn_model.h5              # Model terbaik (checkpoint)
│       └── final_rnn_model.h5             # Model final
│
├── LSTM-Klasifikasi/                      # LSTM & Bi-LSTM untuk Sentiment Analysis
│   ├── lstm.ipynb                         # Implementasi LSTM Unidirectional
│   ├── bi-lstm.ipynb                      # Implementasi Bidirectional LSTM
│   ├── dataset_tweet_sentiment_pilkada_DKI_2017.csv  # Dataset tweet sentiment
│   ├── master_emoji.csv                   # Mapping emoji
│   └── stopword_tweet_pilkada_DKI_2017.csv  # Stopwords Indonesia
│
├── RNN-Klasifikasi/                       # RNN untuk Sentiment Analysis
│   ├── main.ipynb                         # Implementasi Simple RNN
│   ├── dataset_tweet_sentiment_pilkada_DKI_2017.csv  # Dataset tweet sentiment
│   ├── master_emoji.csv                   # Mapping emoji
│   └── stopword_tweet_pilkada_DKI_2017.csv  # Stopwords Indonesia
│
└── README.md                              # Dokumentasi proyek (file ini)
```

---

## 📊 Dataset

### 1. IndoQA Dataset (Question-Answering)
- **Sumber**: Jakarta Artificial Intelligence Research
- **Jenis**: Question-Answering dataset monolingual Bahasa Indonesia
- **Jumlah**: 4,413 contoh (training + validation)
- **Format**: 
  - `context`: Paragraf konteks
  - `question`: Pertanyaan terkait konteks
  - `answer`: Jawaban ekstraktif dari konteks
- **Task**: Extractive Question Answering - memprediksi posisi start dan end dari jawaban dalam konteks

### 2. Dataset Tweet Sentiment Pilkada DKI 2017
- **Sumber**: Tweet sentiment analysis Pilkada DKI Jakarta 2017
- **Jenis**: Sentiment classification (multi-class)
- **Label**: Positive, Neutral, Negative
- **Features**: Text tweets dalam Bahasa Indonesia
- **Task**: Many-to-one sequence classification

---

## 🔧 Prerequisites

Sebelum menjalankan proyek ini, pastikan Anda telah menginstal:

- Python 3.7+
- Jupyter Notebook / JupyterLab
- TensorFlow 2.x
- CUDA (opsional, untuk GPU acceleration)

---

## 📥 Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd Tugas2-RNN-LSTM
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn tqdm
```

Atau gunakan file requirements (jika tersedia):

```bash
pip install -r requirements.txt
```

### 3. Verifikasi TensorFlow

```python
import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## 🚀 Deskripsi Proyek

### 1. RNN & LSTM Chatbot Q&A

#### a. RNN Chatbot (SimpleRNN)
📁 **Lokasi**: `RNN-CHATBOT-INDONESIA-CONTEXT-QUESTION-ANSWER/IPYNB/RNN_Chatbot_QA_Indonesian.ipynb`

**Arsitektur**:
- **Encoder**: SimpleRNN dengan Backpropagation Through Time (BPTT)
- **Input**: Context + Question (diproses secara terpisah)
- **Output**: Prediksi posisi start dan end jawaban
- **Layers**:
  - Embedding Layer
  - SimpleRNN Layer (Bidirectional)
  - Dense Layer untuk start/end prediction

**Karakteristik**:
- ✅ Sederhana dan cepat untuk training
- ⚠️ Rentan terhadap vanishing gradient
- ⚠️ Sulit menangkap long-term dependencies

#### b. LSTM Chatbot
📁 **Lokasi**: `LSTM-CHATBOT/LSTM_Chatbot.ipynb`

**Arsitektur**:
- **Encoder**: LSTM dengan memory cells
- **Gates**: Forget, Input, Output gates untuk mengatur aliran informasi
- **Input**: Context + Question (diproses secara terpisah)
- **Output**: Prediksi posisi start dan end jawaban
- **Layers**:
  - Embedding Layer
  - LSTM Layer (Bidirectional)
  - Dense Layer untuk start/end prediction

**Karakteristik**:
- ✅ Mengatasi vanishing gradient problem
- ✅ Mampu menangkap long-term dependencies
- ✅ Lebih stabil dalam training
- ⚠️ Lebih kompleks dan lambat dibanding SimpleRNN

**Perbandingan RNN vs LSTM untuk Q&A**:

| Aspek | SimpleRNN | LSTM |
|-------|-----------|------|
| **Memory** | Short-term | Long-term |
| **Gradient Problem** | Vanishing gradient | Mengatasi vanishing gradient |
| **Kompleksitas** | Rendah | Tinggi (4 gates) |
| **Training Speed** | Cepat | Lebih lambat |
| **Performa Q&A** | Baik untuk konteks pendek | Baik untuk konteks panjang |

---

### 2. RNN & LSTM Sentiment Classification

#### a. RNN Sentiment Classification
📁 **Lokasi**: `RNN-Klasifikasi/main.ipynb`

**Task**: Many-to-one sequence classification

**Arsitektur**:
```
Input Tweet → Embedding → SimpleRNN → Dense → Softmax (3 classes)
```

**Pipeline**:
1. **Preprocessing**: 
   - Cleaning (remove URL, mention, hashtag)
   - Emoji normalization
   - Stopword removal
2. **Tokenization**: Convert text to sequences
3. **Model**: SimpleRNN dengan dropout
4. **Output**: Positive/Neutral/Negative

**Karakteristik**:
- ✅ Baseline model untuk sentiment analysis
- ✅ Cepat dan efisien
- ⚠️ Performa terbatas pada sequential patterns yang kompleks

#### b. LSTM Sentiment Classification (Unidirectional)
📁 **Lokasi**: `LSTM-Klasifikasi/lstm.ipynb`

**Arsitektur**:
```
Input Tweet → Embedding → LSTM (Forward) → Dense → Softmax (3 classes)
```

**Komponen LSTM**:
1. **Forget Gate**: Menentukan informasi yang dilupakan
   - $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. **Input Gate**: Menentukan informasi baru yang ditambahkan
   - $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
3. **Cell State Update**: Long-term memory
   - $C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$
4. **Output Gate**: Menentukan output/hidden state
   - $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

**Karakteristik**:
- ✅ Menangkap dependencies lebih baik dari SimpleRNN
- ✅ Context dari satu arah (forward)
- ✅ Cocok untuk real-time prediction

#### c. Bidirectional LSTM
📁 **Lokasi**: `LSTM-Klasifikasi/bi-lstm.ipynb`

**Arsitektur**:
```
Input Tweet → Embedding → Bi-LSTM (Forward + Backward) → Dense → Softmax
```

**Karakteristik**:
- ✅ Menangkap context dari kedua arah (past + future)
- ✅ Performa terbaik untuk sentiment analysis
- ✅ Memahami konteks kalimat secara menyeluruh
- ⚠️ 2x parameter dibanding LSTM unidirectional
- ⚠️ Tidak cocok untuk real-time prediction

**Perbandingan RNN vs LSTM vs Bi-LSTM untuk Sentiment**:

| Aspek | SimpleRNN | LSTM | Bi-LSTM |
|-------|-----------|------|---------|
| **Direction** | Forward | Forward | Forward + Backward |
| **Context** | Past only | Past only | Past + Future |
| **Memory** | Short-term | Long-term | Long-term |
| **Parameters** | Rendah | Sedang | Tinggi |
| **Accuracy** | Baseline | Lebih baik | Terbaik |
| **Training Time** | Cepat | Sedang | Lambat |
| **Real-time** | ✅ Ya | ✅ Ya | ❌ Tidak |

---

## 📈 Hasil dan Model

### Model yang Dihasilkan

#### Chatbot Q&A Models:
1. **RNN Models** (di `RNN-CHATBOT-INDONESIA-CONTEXT-QUESTION-ANSWER/Output Model RNN/`):
   - `best_rnn_model.h5` - Model dengan validation loss terendah
   - `final_rnn_model.h5` - Model setelah training lengkap

2. **LSTM Models** (di `LSTM-CHATBOT/Output Model LSTM/`):
   - `best_lstm_model.h5` - Model dengan validation loss terendah
   - `final_lstm_model.h5` - Model setelah training lengkap

### Metrics
- **Q&A Task**: Exact Match (EM), F1-Score
- **Sentiment Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## ▶️ Cara Menjalankan

### 1. RNN Chatbot Q&A
```bash
cd "RNN-CHATBOT-INDONESIA-CONTEXT-QUESTION-ANSWER/IPYNB"
jupyter notebook RNN_Chatbot_QA_Indonesian.ipynb
```
Jalankan semua cells secara berurutan.

### 2. LSTM Chatbot Q&A
```bash
cd "LSTM-CHATBOT"
jupyter notebook LSTM_Chatbot.ipynb
```
Jalankan semua cells secara berurutan.

### 3. RNN Sentiment Classification
```bash
cd "RNN-Klasifikasi"
jupyter notebook main.ipynb
```
Jalankan semua cells secara berurutan.

### 4. LSTM Sentiment Classification
```bash
cd "LSTM-Klasifikasi"
jupyter notebook lstm.ipynb
```
Jalankan semua cells secara berurutan.

### 5. Bi-LSTM Sentiment Classification
```bash
cd "LSTM-Klasifikasi"
jupyter notebook bi-lstm.ipynb
```
Jalankan semua cells secara berurutan.

---

## 🛠 Teknologi yang Digunakan

### Framework & Libraries
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Komputasi numerik
- **Pandas**: Data manipulation dan analysis
- **Matplotlib & Seaborn**: Visualisasi data
- **Scikit-learn**: Preprocessing dan evaluasi
- **TQDM**: Progress bar

### Model Architectures
- **SimpleRNN**: Recurrent Neural Network dasar
- **LSTM**: Long Short-Term Memory
- **Bidirectional LSTM**: LSTM dengan context dua arah
- **Embedding Layer**: Word embeddings
- **Dense Layer**: Fully connected layer

### Techniques
- **Backpropagation Through Time (BPTT)**: Training RNN
- **Word Tokenization**: Text preprocessing
- **Padding**: Sequence normalization
- **Dropout**: Regularization
- **Early Stopping**: Prevent overfitting
- **Model Checkpoint**: Save best model

---

## 🔍 Catatan Penting

### Tips untuk Training:
1. **GPU Acceleration**: Gunakan GPU jika tersedia untuk mempercepat training
2. **Batch Size**: Sesuaikan dengan memory GPU/RAM yang tersedia
3. **Learning Rate**: Gunakan ReduceLROnPlateau untuk adaptive learning rate
4. **Early Stopping**: Patience 3-5 epochs untuk mencegah overfitting

### Troubleshooting:
- **Out of Memory**: Kurangi batch size atau max sequence length
- **Slow Training**: Gunakan GPU atau kurangi kompleksitas model
- **Poor Performance**: Tambah epochs atau tuning hyperparameters

---

