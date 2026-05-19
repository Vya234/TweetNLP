# TweetNLP Sentiment Analysis from Scratch

This project implements tweet sentiment classification using deep learning models built entirely from scratch in PyTorch, without using any pretrained language models.

The objective is to study how architectural improvements such as Bidirectional LSTMs and attention mechanisms affect sentiment classification performance on noisy social media text.

This work is inspired by the paper *TweetNLP: Cutting-Edge Natural Language Processing for Social Media* and uses the TweetEval sentiment dataset.

---

## Problem Statement

Given a tweet, the model predicts one of the following sentiment classes:

- Negative
- Neutral
- Positive

### Example Predictions

| Input Tweet | Predicted Sentiment |
|----------|----------------|
| "I love this product!" | Positive |
| "This is terrible." | Negative |
| "It is okay." | Neutral |

---

## Dataset

This project uses the TweetEval sentiment analysis dataset, derived from SemEval 2017 Task 4.

### Dataset Size

| Split | Number of Tweets |
|------:|-----------------:|
| Train | ~45,000 |
| Validation | ~2,000 |
| Test | ~12,000 |

### Classes

- Negative
- Neutral
- Positive

---

## Project Structure

```text
TweetNLP/
│
├── dataset/
│   ├── train_text.txt
│   ├── train_labels.txt
│   ├── val_text.txt
│   ├── val_labels.txt
│   ├── test_text.txt
│   └── test_labels.txt
│
├── baseline_model.py      # Baseline LSTM model
├── improved_model.py      # BiLSTM with Attention
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── utils.py               # Data loading and preprocessing
└── README.md
```

---

## Data Preprocessing Pipeline

Before training, raw tweets are converted into numerical form using the following steps:

1. Load tweet text and labels
2. Tokenize each tweet into words
3. Build a vocabulary from the training data
4. Convert each word to a unique integer index
5. Encode tweets as sequences of integers
6. Pad all sequences to a fixed maximum length
7. Convert sequences into PyTorch tensors

### Pipeline

```text
Tweet → Tokenization → Vocabulary Mapping → Encoding → Padding → Tensor Conversion → Model
```

---

## Model Architectures

### Baseline Model

```text
Embedding → LSTM → Fully Connected Layer
```

### Improved Model

```text
Embedding → Bidirectional LSTM → Attention → Dropout → Fully Connected Layer
```

### Improvements Over Baseline

- **Bidirectional LSTM (BiLSTM):** Captures context from both directions.
- **Attention Mechanism:** Focuses on the most informative words in a tweet.
- **Dropout:** Reduces overfitting and improves generalization.

---

## Training Configuration

| Parameter | Value |
|---------:|------:|
| Embedding Dimension | 100 |
| Hidden Dimension | 128 |
| Batch Size | 64 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Confusion Matrix

### Why Macro F1 Score?

The dataset is imbalanced, with the neutral class containing more samples than the other classes. Macro F1 Score computes the F1 score for each class individually and averages them, giving equal importance to all classes.

---

## Results

| Model | Accuracy | Macro F1 Score |
|------|---------:|---------------:|
| Baseline (LSTM) | ~0.627 | ~0.5909 |
| BiLSTM | ~0.628 | ~0.5936 |
| BiLSTM + Attention | ~0.626 | ~0.5975 |

---

## Confusion Matrix Analysis

The confusion matrix reveals the following insights:

- The model performs best on the neutral and positive classes.
- Negative tweets are frequently misclassified as neutral.
- Neutral sentiment acts as a boundary class between positive and negative sentiment.

The attention mechanism improves contextual understanding by allowing the model to focus on important sentiment-bearing words.

---

## How to Run

### Train the Model

```bash
python3 train.py
```

### Evaluate the Model

```bash
python3 evaluate.py
```

---

## Model Selection

In both `train.py` and `evaluate.py`, choose the model to run:

```python
MODEL_TYPE = "baseline"
```

or

```python
MODEL_TYPE = "improved"
```

---

## Limitations

- No pretrained models were used.
- Performance is lower than transformer-based approaches reported in the original TweetNLP paper.
- The project focuses only on sentiment analysis.

---

## Conclusion

This project demonstrates that architectural improvements such as Bidirectional LSTMs and attention mechanisms can improve tweet sentiment classification performance even when models are implemented entirely from scratch.

It also highlights the challenges of sentiment analysis on noisy social media text, particularly the difficulty of distinguishing neutral sentiment from positive and negative sentiment.

---

## References

1. Camacho-Collados, J., et al. (2022). *TweetNLP: Cutting-Edge Natural Language Processing for Social Media*. https://arxiv.org/abs/2206.14774
2. TweetEval Dataset. https://github.com/cardiffnlp/tweeteval

---

## Author

**Kavya Rai**  
IIT Kharagpur
