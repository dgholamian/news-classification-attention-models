# News Classification with LSTM + Attention and GRU + Attention

This project demonstrates a full pipeline for text classification on the [AG News dataset](https://www.tensorflow.org/datasets/catalog/ag_news_subset) using TensorFlow. It includes preprocessing, visualization, and training two deep learning models: one with LSTM + Attention and another with GRU + Attention.

---

## 📊 Dataset

- **Source:** AG News Subset via `tensorflow_datasets`
- **Categories:** World, Sports, Business, Sci/Tech
- **Samples:** 120,000 training / 7,600 testing

---

## 🧹 Preprocessing Pipeline

1. **Lowercasing**
2. **Removing punctuation/numbers**
3. **Stopword removal using NLTK**
4. **Tokenization using `TextVectorization`**
5. **One-hot encoding of labels**

---

## 🌈 Visualization

- **Word Clouds:** Top words per class
- **Label Distribution:** Frequency of each class

Sample plots:
- `training_wordclouds.png`
- `test_wordclouds.png`
- `training_dist.png`

---

## 🧠 Model Architectures

### 🔸 LSTM + Attention

- Embedding layer
- Bidirectional LSTM
- Scaled Dot-Product Attention
- Global Average Pooling
- Dense softmax layer

### 🔸 GRU + Attention

- Same as above, but using Bidirectional GRU

---

## 🛠 Training Setup

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Epochs:** 10
- **Early Stopping:** Based on validation loss

---

## 📈 Results

### Accuracy & Loss Plots
- `lstm_attention.png`
- `gru_attention.png`

### Confusion Matrices
- `Confusion_matrix_LSTM+Attention.png`
- `Confusion_matrix_GRU+Attention.png`

---

## 💾 Model Files

- `lstm_attention_model.keras`, `lstm_attention.weights.h5`
- `gru_attention_model.keras`, `gru_attention.weights.h5`

---


