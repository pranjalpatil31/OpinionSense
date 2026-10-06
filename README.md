# 💬 OpinionSense: LSTM-Based Sentiment Analysis and Classification of Social Media Text

**OpinionSense** is a Deep Learning-based sentiment analysis system that classifies social media text into three sentiment categories:

* 😊 **Positive**
* 😞 **Negative**
* 😐 **Neutral**

The project uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network with **pretrained GloVe word embeddings** to understand the sequential and contextual information present in text.

A **Gradio web interface** is also provided for real-time prediction and batch CSV sentiment analysis.

---

## 📌 Project Overview

Social media platforms contain a large amount of textual information expressing people's opinions, emotions, reactions, and experiences. Automatically identifying the sentiment of such text is an important Natural Language Processing (NLP) task.

OpinionSense uses a Deep Learning approach to analyze a text input and predict its sentiment.

The complete processing pipeline is:

```text
Social Media Text
        ↓
Text Cleaning & Normalization
        ↓
Duplicate Removal
        ↓
Stratified Train / Validation / Test Split
        ↓
Word Tokenization
        ↓
Integer Encoding
        ↓
Sequence Padding
        ↓
Pretrained GloVe Embeddings
        ↓
Bidirectional LSTM
        ↓
Dense Layer
        ↓
Softmax Output
        ↓
Positive / Negative / Neutral
```

---

## 🎯 Objectives

The main objectives of OpinionSense are:

1. To develop a Deep Learning-based sentiment classification system.
2. To apply an **LSTM-based architecture** for processing sequential text data.
3. To use a **Bidirectional LSTM** to capture contextual information from both directions of a sentence.
4. To preprocess and normalize social media text before classification.
5. To use pretrained **GloVe word embeddings** for better word representation.
6. To classify text into Positive, Negative, and Neutral categories.
7. To evaluate the model using accuracy, loss, Macro F1-score, classification report, and confusion matrix.
8. To provide a simple web interface for real-time sentiment prediction.
9. To support batch sentiment prediction using CSV files.

---

## 🧠 Deep Learning Approach

The core of OpinionSense is a **Bidirectional LSTM neural network**.

LSTM is a type of Recurrent Neural Network (RNN) designed to process sequential data while reducing the difficulty of learning long-term dependencies.

For text classification, the words in a sentence form a sequence. The LSTM processes this sequence and learns patterns that are useful for determining sentiment.

### Why Bidirectional LSTM?

A normal LSTM processes a sequence primarily in one direction.

A Bidirectional LSTM processes the sequence in both directions:

```text
Forward:
I → really → love → this → product

Backward:
product → this → love → really → I
```

This allows the network to use information from both previous and following words when creating the representation of the text.

For sentiment analysis, this can be useful because the meaning of a word may depend on surrounding words.

---

# 🏗️ Model Architecture

The implemented model follows this architecture:

```text
Input Sequence
     │
     ▼
Embedding Layer
     │
     ├── 100-dimensional word vectors
     ├── GloVe pretrained embeddings when available
     └── Masking enabled
     │
     ▼
Dropout (0.30)
     │
     ▼
Bidirectional LSTM (64 units)
     │
     ▼
Dropout (0.40)
     │
     ▼
Dense Layer (64 units, ReLU)
     │
     ▼
Dropout (0.30)
     │
     ▼
Dense Layer (3 units, Softmax)
     │
     ▼
Sentiment Prediction
```

### Model Layers

| Layer              | Configuration        | Purpose                                  |
| ------------------ | -------------------- | ---------------------------------------- |
| Input              | Sequence length = 40 | Accepts padded text sequences            |
| Embedding          | 100 dimensions       | Converts word IDs into numerical vectors |
| Dropout            | 30%                  | Reduces overfitting                      |
| Bidirectional LSTM | 64 units             | Learns contextual sequential patterns    |
| Dropout            | 40%                  | Regularization                           |
| Dense              | 64 neurons, ReLU     | Learns higher-level features             |
| Dropout            | 30%                  | Regularization                           |
| Output Dense       | 3 neurons, Softmax   | Produces three sentiment probabilities   |

---

# 📂 Dataset

The notebook expects a CSV file named:

```text
projectML.csv
```

The dataset must contain at least the following columns:

| Column      | Description               |
| ----------- | ------------------------- |
| `Text`      | Social media text/post    |
| `Sentiment` | Numerical sentiment label |

The implemented label mapping is:

```text
0 → Positive
1 → Negative
2 → Neutral
```

Example:

```csv
Text,Sentiment
"I love this product!",0
"This is terrible.",1
"Going to the conference today.",2
```

> **Note:** The dataset itself is not generated by the notebook. You must provide `projectML.csv` when running the project.

---

# 🔄 Complete Data Processing Pipeline

## 1. Dataset Loading

The notebook loads `projectML.csv`.

The implementation checks that the required columns exist:

```text
Text
Sentiment
```

The CSV is loaded using Latin-1 encoding to handle the encoding characteristics of the supplied dataset.

---

## 2. Dataset Audit

Before training, the notebook checks:

* Missing values
* Duplicate text entries
* Sentiment label distribution
* Dataset shape
* Class distribution

A sentiment distribution chart is also generated.

---

## 3. Text Cleaning

Raw social media text can contain URLs, mentions, hashtags, HTML, contractions, emojis, punctuation, and encoding artifacts.

OpinionSense applies the following cleaning operations:

```text
Raw Text
   ↓
Encoding Repair
   ↓
Lowercase Conversion
   ↓
URL Removal
   ↓
HTML Removal
   ↓
Mention Removal
   ↓
Hashtag Symbol Removal
   ↓
Contraction Expansion
   ↓
Special Character Removal
   ↓
Whitespace Normalization
   ↓
Clean Text
```

### Examples

```text
"Check this out! https://example.com"
                ↓
"check this out"
```

```text
"@user #AmazingProduct"
                ↓
"amazingproduct"
```

The system also preserves negation during contraction expansion.

For example:

```text
"can't"
   ↓
"can not"
```

This is important because negation can change sentiment.

---

# ✂️ Duplicate Removal

After cleaning, duplicate text entries are removed.

This prevents the same text from appearing multiple times and potentially being present in both training and testing data.

The notebook performs:

```python
data.drop_duplicates(subset="clean_text")
```

---

# 📊 Train / Validation / Test Split

The dataset is divided using a **stratified split**.

The final proportions are:

```text
70% → Training
15% → Validation
15% → Testing
```

Diagram:

```text
                Complete Dataset
                       │
                       ▼
              ┌────────────────┐
              │  Stratified    │
              │     Split      │
              └────────────────┘
                 /      |      \
                /       |       \
               ▼        ▼        ▼
           Training  Validation  Testing
             70%       15%        15%
```

Stratification ensures that the three sentiment classes remain proportionally represented across the splits.

The test set is kept separate and is used for the final evaluation.

---

# 🔤 Tokenization and Integer Encoding

Neural networks cannot directly process raw text.

Therefore, OpinionSense converts words into numerical IDs.

A custom `SimpleTokenizer` is implemented in the notebook.

Two special tokens are used:

```text
0 → <PAD>
1 → <OOV>
```

Where:

* `<PAD>` represents padding.
* `<OOV>` represents an unknown word.

Example:

```text
"i love this movie"
```

can be converted into something similar to:

```text
[25, 73, 14, 92]
```

The actual numbers depend on the generated vocabulary.

---

# 📏 Sequence Padding

The model expects sequences of a fixed length.

The notebook uses:

```text
MAX_LENGTH = 40
```

Shorter sequences are padded with `0`.

Example:

```text
[25, 73, 14]
```

becomes:

```text
[25, 73, 14, 0, 0, 0, ...]
```

up to 40 positions.

Longer sequences are truncated to the maximum sequence length.

---

# 🧠 GloVe Word Embeddings

OpinionSense uses **100-dimensional GloVe embeddings** when the GloVe file is available.

The configured embedding is:

```text
GloVe 6B
Dimension: 100
```

GloVe provides pretrained numerical representations of words.

Instead of representing a word only as an arbitrary ID, the embedding represents the word using a vector containing semantic information.

For example, words with related meanings can have similar vector representations.

The notebook downloads:

```text
glove.6B.zip
```

when the required embedding file is not already present.

The embedding matrix is then created according to the project's vocabulary.

### Small Dataset Consideration

The supplied dataset is relatively small, so pretrained embeddings are useful because the model does not have to learn every word representation entirely from the training data.

---

# 🔁 GloVe Fallback

The notebook also contains a fallback mechanism.

If GloVe cannot be downloaded or loaded:

```text
GloVe unavailable
       ↓
Trainable random embedding
```

In that situation, the embedding layer becomes trainable.

When GloVe is successfully loaded, the implementation uses the pretrained embeddings as a frozen embedding representation.

---

# 🧠 Bidirectional LSTM

The main Deep Learning component is:

```python
Bidirectional(LSTM(64))
```

The LSTM receives the padded word sequence after embedding.

The Bidirectional wrapper allows the model to process the sequence in both forward and backward directions.

The resulting representation is then passed to:

```text
Dropout
   ↓
Dense(64, ReLU)
   ↓
Dropout
   ↓
Dense(3, Softmax)
```

---

# 🎯 Output Layer

The final layer contains three neurons:

```python
Dense(3, activation="softmax")
```

The three outputs correspond to:

```text
Positive
Negative
Neutral
```

Softmax converts the outputs into probabilities.

For example, a prediction could conceptually look like:

```text
Positive : 0.82
Negative : 0.06
Neutral  : 0.12
```

The class with the highest probability becomes the predicted sentiment.

---

# ⚙️ Model Training

The model is compiled using:

```text
Optimizer:
Adam

Learning Rate:
0.001

Loss:
Sparse Categorical Crossentropy

Metric:
Accuracy
```

Training configuration:

```text
Maximum Epochs = 80
Batch Size = 16
```

The model does not necessarily run all 80 epochs because Early Stopping is used.

---

# ⚖️ Class Weighting

The notebook calculates class weights using:

```python
compute_class_weight("balanced", ...)
```

This helps compensate for differences in the number of examples belonging to each sentiment class.

The calculated weights are supplied during training:

```python
class_weight=class_weight
```

This makes the training process less biased toward classes with more samples.

---

# 🛑 Early Stopping

The model uses:

```text
EarlyStopping
```

with:

```text
monitor = validation loss
patience = 10
restore_best_weights = True
```

This means that if validation loss does not improve for several epochs, training stops and the best validation weights are restored.

---

# 📉 Learning Rate Reduction

The notebook also uses:

```text
ReduceLROnPlateau
```

Configuration:

```text
monitor = validation loss
factor = 0.5
patience = 3
minimum learning rate = 1e-5
```

If validation loss stops improving, the learning rate is reduced to allow finer optimization.

---

# 📈 Training Visualization

During training, the notebook plots:

### Training vs Validation Accuracy

```text
Epoch → Accuracy
```

### Training vs Validation Loss

```text
Epoch → Loss
```

These plots help observe whether the model is learning effectively and whether signs of overfitting appear.

---

# 🧪 Model Evaluation

After training, the model is evaluated using the untouched test set.

The notebook calculates:

* Test Accuracy
* Test Loss
* Macro F1-score
* Precision
* Recall
* F1-score for each class
* Confusion Matrix

The test set is not used for model fitting.

---

# 📊 Classification Report

The notebook generates a classification report containing:

```text
              precision
              recall
              f1-score
              support
```

for:

```text
Positive
Negative
Neutral
```

The **Macro F1-score** is also calculated.

Macro F1 gives equal importance to each class by averaging the F1-score across the three sentiment categories.

---

# 🔲 Confusion Matrix

A confusion matrix is generated to understand the classification behaviour of the model.

It compares:

```text
Actual Sentiment
       vs
Predicted Sentiment
```

The matrix helps identify which sentiment categories are correctly classified and where the model makes mistakes.

The generated image is saved as:

```text
confusion_matrix.png
```

---

# 🔍 Error Analysis

The notebook identifies incorrectly classified test samples.

For each misclassified sample, it displays:

```text
Text
Actual Sentiment
Predicted Sentiment
Confidence
```

This allows the model's errors to be manually inspected.

Error analysis can help identify difficult cases such as:

* Ambiguous statements
* Context-dependent expressions
* Short social media posts
* Mixed sentiment
* Unusual wording

---

# 🔮 Sentiment Prediction

A reusable prediction function is implemented:

```python
predict_sentiment(text)
```

The function performs the same preprocessing pipeline used during training:

```text
Input Text
    ↓
Cleaning
    ↓
Vocabulary Encoding
    ↓
Padding
    ↓
Trained BiLSTM
    ↓
Probability Prediction
    ↓
Final Sentiment
```

The function returns:

```text
sentiment
confidence
probabilities
```

Example structure:

```python
{
    "sentiment": "Positive",
    "confidence": 0.91,
    "probabilities": {
        "Positive": 0.91,
        "Negative": 0.03,
        "Neutral": 0.06
    }
}
```

The actual values depend on the trained model.

---

# 🧪 Sanity Testing

The notebook also tests the trained model on several new sentences that were not part of the training process.

Examples include:

```text
"I love this so much, best day ever!"
```

```text
"This is terrible, I hate it"
```

```text
"Going with the flow."
```

The predicted class is compared with an expected sentiment label for each example.

This provides a simple sanity check of the trained model's behaviour on new text.

---

# 💾 Saved Model Files

After training, the notebook saves two important files.

### 1. Trained Model

```text
opinionsense_bilstm.keras
```

This contains the trained neural network.

### 2. Vocabulary Configuration

```text
opinionsense_vocab.json
```

This stores:

* Word-to-index mapping
* Maximum sequence length
* Label mapping

These files can be used to preserve the trained model and its text-processing configuration.

---

# 🌐 Gradio Web Application

OpinionSense includes a Gradio-based frontend.

The application contains three main tabs:

```text
┌──────────────────────────────────────────┐
│             OpinionSense                 │
├──────────────┬───────────┬───────────────┤
│   Analyze    │   Batch   │ Model Report  │
└──────────────┴───────────┴───────────────┘
```

---

## 1. Analyze Tab

The Analyze tab allows the user to enter a social media post.

Example:

```text
Just got the job offer, I can't stop smiling today!
```

After clicking **Analyze**, the system displays:

* Predicted sentiment
* Confidence
* Probability for each class
* Recent prediction history

The interface also provides example inputs.

---

## 2. Batch CSV Tab

The Batch tab allows multiple text records to be classified at once.

The uploaded CSV should contain:

```text
Text
```

If the `Text` column is not available, the first column is used.

The system generates:

```text
Predicted_Sentiment
Confidence
```

and saves the result as:

```text
opinionsense_predictions.csv
```

The generated CSV can then be downloaded from the interface.

---

## 3. Model Report Tab

The Model Report tab displays the final test-set information, including:

* Accuracy
* Macro F1-score
* Number of test samples
* Model architecture summary
* Training sample count
* Vocabulary size
* Confusion matrix

The values shown are generated dynamically from the actual training run.

---

# 🎨 User Interface

The Gradio application provides:

* Text input box
* Analyze button
* Clear button
* Example inputs
* Sentiment probability visualization
* Prediction history
* CSV upload
* Batch prediction
* Downloadable prediction file
* Model performance report
* Confusion matrix

The interface is designed for demonstration during project evaluation or viva.

---

# 🛠️ Technologies Used

| Technology         | Purpose                                      |
| ------------------ | -------------------------------------------- |
| Python             | Main programming language                    |
| TensorFlow / Keras | Deep Learning model                          |
| LSTM               | Sequential text modelling                    |
| Bidirectional LSTM | Forward and backward contextual processing   |
| GloVe              | Pretrained word embeddings                   |
| NumPy              | Numerical operations                         |
| Pandas             | Dataset processing                           |
| Scikit-learn       | Data splitting, class weights and evaluation |
| Matplotlib         | Visualization                                |
| Seaborn            | Confusion matrix visualization               |
| Gradio             | Web-based frontend                           |
| Google Colab       | Recommended execution environment            |

---

# 📦 Python Libraries

The main libraries used by the notebook are:

```text
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
gradio
```

Gradio is installed in the notebook using:

```bash
pip install -U gradio
```

---

# 🚀 Running the Project in Google Colab

Google Colab is recommended because the notebook is designed to work directly with the Colab environment.

## Step 1: Open the Notebook

Upload/open:

```text
OpinionSense_LSTM_Final.ipynb
```

in Google Colab.

---

## Step 2: Install Gradio

Run the first code cell:

```python
!pip -q install -U gradio
```

---

## Step 3: Upload Dataset

The notebook expects:

```text
projectML.csv
```

If the file is not already available in the Colab working directory, the notebook automatically opens a file upload dialog.

Upload:

```text
projectML.csv
```

---

## Step 4: Run the Notebook

Run the cells from top to bottom.

The notebook performs:

```text
Dataset Loading
      ↓
Dataset Audit
      ↓
Text Cleaning
      ↓
Duplicate Removal
      ↓
Train/Validation/Test Split
      ↓
Tokenization
      ↓
Padding
      ↓
GloVe Loading
      ↓
BiLSTM Construction
      ↓
Model Training
      ↓
Evaluation
      ↓
Error Analysis
      ↓
New Text Prediction
      ↓
Model Saving
      ↓
Gradio Application
```

---

# 🌐 Launching the Web Application

The final cell launches the Gradio application:

```python
demo.launch(share=True, debug=False)
```

Because `share=True` is enabled, Gradio generates a public sharing link.

The link can be opened in a browser and used to demonstrate the OpinionSense application.

---

# 📁 Expected Project Structure

A practical project structure can be:

```text
OpinionSense/
│
├── OpinionSense_LSTM_Final.ipynb
├── projectML.csv
│
├── glove.6B.100d.txt
│
├── opinionsense_bilstm.keras
├── opinionsense_vocab.json
├── confusion_matrix.png
│
├── opinionsense_predictions.csv
│
└── README.md
```

The generated files such as the trained model, confusion matrix, and prediction CSV are produced after running the notebook.

---

# 🔬 Technical Workflow

The complete technical workflow can be summarized as follows:

### Phase 1 — Data Preparation

```text
projectML.csv
      ↓
Required Column Validation
      ↓
Missing Value Handling
      ↓
Label Validation
      ↓
Text Cleaning
      ↓
Duplicate Removal
```

### Phase 2 — Data Splitting

```text
Clean Dataset
      ↓
Stratified Split
      ↓
70% Training
15% Validation
15% Testing
```

### Phase 3 — Text Representation

```text
Clean Text
      ↓
Vocabulary
      ↓
Integer Encoding
      ↓
Padding to 40 tokens
      ↓
100-dimensional Embedding
```

### Phase 4 — Deep Learning

```text
Embedding
    ↓
Dropout
    ↓
Bidirectional LSTM (64)
    ↓
Dropout
    ↓
Dense (64, ReLU)
    ↓
Dropout
    ↓
Softmax (3)
```

### Phase 5 — Training

```text
Class Weights
      ↓
Adam Optimizer
      ↓
Sparse Categorical Crossentropy
      ↓
Early Stopping
      ↓
Learning Rate Reduction
```

### Phase 6 — Evaluation

```text
Test Set
   ↓
Predictions
   ↓
Accuracy
   ↓
Macro F1
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Error Analysis
```

### Phase 7 — Deployment

```text
Trained BiLSTM
      ↓
Prediction Function
      ↓
Gradio Interface
      ↓
Analyze / Batch / Model Report
```

---

# 📌 Important Implementation Details

The current implementation uses the following configuration:

```text
Number of classes       : 3
Maximum sequence length: 40
Embedding dimension    : 100
LSTM units             : 64
Dense units            : 64
Maximum epochs         : 80
Batch size             : 16
Optimizer              : Adam
Learning rate          : 0.001
Loss                   : Sparse Categorical Crossentropy
```

Regularization is applied using:

```text
Dropout
L2 regularization
Early stopping
```

---

# ⚠️ Important Notes

### Dataset

The project requires the supplied:

```text
projectML.csv
```

dataset.

The README does not assume a particular dataset size because the final number of rows depends on the actual CSV after cleaning and duplicate removal.

### Model Results

Accuracy, F1-score, loss, confusion matrix values, vocabulary size, and number of training epochs are **not hard-coded** in this README.

They are generated by the notebook during execution.

This is important because the results can change if:

* The dataset changes
* Duplicate records change
* GloVe availability changes
* TensorFlow versions change
* Training conditions change

### GloVe

The notebook attempts to download GloVe automatically.

The download is approximately 800 MB for the ZIP archive.

If the download fails, the implementation automatically falls back to a trainable random embedding.

### Confidence

The displayed confidence is the highest softmax probability produced by the model.

It should be interpreted as the model's prediction confidence, **not as a guaranteed probability that the sentiment is objectively correct**.

---

# ⚠️ Limitations

The current implementation has several limitations:

1. The model is trained specifically for the sentiment classes represented in the supplied dataset.
2. It may perform poorly on text that is substantially different from the training data.
3. Sarcasm and irony can be difficult to classify.
4. Very short or ambiguous text may produce uncertain predictions.
5. Mixed sentiment in a single sentence may be difficult to classify.
6. Social media language changes over time, which can affect model performance.
7. The dataset size is relatively small compared with large-scale NLP datasets.
8. GloVe provides general-purpose word representations and is not specifically trained for this sentiment dataset.
9. The model's confidence should not be interpreted as a guarantee of correctness.

---

# 🔐 Responsible Use

OpinionSense is intended for:

* Academic projects
* NLP experimentation
* Sentiment analysis demonstrations
* Educational purposes
* Research and model evaluation

The predictions should not be treated as definitive assessments of a person's emotions, mental state, or personality.

---

# 🔮 Possible Future Enhancements

Future versions of OpinionSense could include:

* Larger and more diverse sentiment datasets
* Transformer-based models such as BERT
* Attention mechanisms
* Context-aware sentiment analysis
* Multilingual sentiment classification
* Aspect-based sentiment analysis
* Sarcasm detection
* Emotion classification
* Model comparison between LSTM, GRU, BiLSTM and Transformers
* Cloud deployment
* REST API deployment
* User authentication
* Persistent prediction history
* Real-time social media data integration

---

# 📊 Project Summary

OpinionSense demonstrates an end-to-end **Deep Learning NLP pipeline** for sentiment classification.

The system starts with raw social media text and performs preprocessing, tokenization, numerical encoding, padding, pretrained word embedding, and sequential modelling using a **Bidirectional LSTM**.

The trained model predicts one of three sentiment classes:

```text
Positive
Negative
Neutral
```

The system also evaluates the model on a held-out test set and provides accuracy, Macro F1-score, classification metrics, and a confusion matrix.

Finally, the trained model is integrated with a **Gradio web application**, allowing users to perform individual predictions as well as batch CSV sentiment analysis.

---

# 🧾 One-Line Project Description

> **OpinionSense is a Bidirectional LSTM-based Deep Learning system that analyzes social media text and classifies it as Positive, Negative, or Neutral using text preprocessing, pretrained GloVe embeddings, and a Gradio-based web interface.**

---

# 👩‍💻 Project

**Project Name:** OpinionSense
**Project Type:** Deep Learning / Natural Language Processing
**Task:** Sentiment Analysis and Classification
**Model:** Bidirectional LSTM
**Embedding:** GloVe 100-dimensional embeddings
**Classes:** Positive, Negative, Neutral
**Frontend:** Gradio
**Recommended Environment:** Google Colab
**Language:** Python
