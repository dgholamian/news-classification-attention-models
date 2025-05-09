# Hybrid of LSTM+Attention Layer and GRU+Attention Layer for News Classification

This project demonstrates a full pipeline for text classification on the [AG News dataset](https://www.tensorflow.org/datasets/catalog/ag_news_subset) using TensorFlow. It includes preprocessing, visualization, and training two deep learning models: one with LSTM + Attention layer and another with GRU + Attention layer.

---

## Dataset

- **Source:** AG News Subset via `tensorflow_datasets`
- **Categories:** World, Sports, Business, Sci/Tech
- **Samples:** 120,000 training / 7,600 testing

**Word Cloud for original training data**
![Training Dataset Word Cloud](./Figures/training_wordclouds.png "Training Dataset Word Cloud")


**Word Cloud for original test data**
![Test Dataset Word Cloud](./Figures/test_wordclouds.png "Test Dataset Word Cloud")
---

## Preprocessing Pipeline

1. **Lowercasing**
2. **Removing punctuation/numbers**
3. **Stopword removal using NLTK**
4. **Tokenization using `TextVectorization`**
5. **One-hot encoding of labels**

   **Word Cloud for cleaned training data**
![Cleaned Training Dataset Word Cloud](./Figures/cleaned_training_wordclouds.png "Cleaned Training Dataset Word Cloud")

**Word Cloud for cleaned test data**
![Cleaned Test Dataset Word Cloud](./Figures/cleaned_test_wordclouds.png "Cleaned Test Dataset Word Cloud")

---

## Visualization
### Label Distribution: Frequency of each class

 **Label Distribution Plot for Training Dataset**
![Label Distribution Plot for Training Dataset](./Figures/training_dist.png "Label Distribution Plot for Training Dataset")

**Label Distribution Plot for Test Dataset**
![Label Distribution Plot for Test Dataset](./Figures/test_dist.png "Label Distribution Plot for Test Dataset")

---

##  Model Architectures

### LSTM with Attention Mechanism: Architecture

- Embedding layer
- Bidirectional LSTM
- Scaled Dot-Product Attention
- Global Average Pooling
- Dense softmax layer

### GRU with Attention Mechanism: Architecture

- Embedding layer
- Bidirectional GRU
- Scaled Dot-Product Attention
- Global Average Pooling
- Dense softmax layer

---

## Training Setup

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Epochs:** 10
- **Early Stopping:** Based on validation loss

---

## Results

### Accuracy & Loss Plots
**Confusion Matrix for LSTM with Attention Mechanism**
![Accuracy and Loss plots for LSTM with Attention Mechanism](./Figures/lstm_attention_acc_loss.png "Accuracy and Loss plots for LSTM with Attention Mechanism")


**Confusion Matrix for GRU with Attention Mechanism**
![Accuracy and Loss plots for GRU with Attention Mechanism](./Figures/gru_attention_acc_loss.png "Accuracy and Loss plots for GRU with Attention Mechanism")



### Confusion Matrices
**Confusion Matrix for LSTM with Attention Mechanism**
![Confusion Matrix for LSTM with Attention Mechanism](./Figures/Confusion_matrix_LSTM+Attention.png "Confusion Matrix for LSTM with Attention Mechanism")


**Confusion Matrix for GRU with Attention Mechanism**
![Confusion Matrix for GRU with Attention Mechanism](./Figures/Confusion_matrix_GRU+Attention.png "Confusion Matrix for GRU with Attention Mechanism")


---

## Model Files

- `lstm_attention_model.keras`, `lstm_attention.weights.h5`
- `gru_attention_model.keras`, `gru_attention.weights.h5`

---


