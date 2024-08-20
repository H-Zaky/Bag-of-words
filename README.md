# Sentiment Analysis on IMDB Reviews using Word2Vec and LSTM

This project performs sentiment analysis on movie reviews from the IMDB dataset using a deep learning model that combines Word2Vec embeddings and an LSTM-based neural network. The goal is to classify movie reviews as positive or negative.

## Project Overview

- **Dataset**: The project utilizes multiple datasets of labeled and unlabeled IMDB movie reviews.
- **Text Preprocessing**: The reviews are tokenized, cleaned, and converted into sequences suitable for training deep learning models.
- **Word2Vec Embeddings**: Word2Vec is used to create word embeddings that represent words in a continuous vector space.
- **Model Architecture**: The model is built using a bidirectional LSTM network with a dense layer and dropout regularization.
- **Training**: The model is trained on the IMDB dataset with validation using a portion of the training data.
- **Evaluation**: The model's performance is evaluated using AUC-ROC, and predictions are made on the test set for submission.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Keras
- TensorFlow
- Gensim
- scikit-learn

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the datasets:**
   - The project uses the **IMDB** dataset and the **Word2Vec NLP Tutorial** dataset. Download these datasets and place them in the `input` directory with the following structure:
   
    ```
    input/
    ├── word2vec-nlp-tutorial/
    │   ├── labeledTrainData.tsv.zip
    │   ├── unlabeledTrainData.tsv.zip
    │   └── testData.tsv.zip
    └── imdb-review-dataset/
        └── imdb_master.csv
    ```

## Running the Project

1. **Text Preprocessing:**
   - The reviews are tokenized, converted to lowercase, and cleaned by removing punctuation and stop words.

2. **Word2Vec Embedding:**
   - Word2Vec is used to train word embeddings on the cleaned review data. The resulting embeddings are used as the input for the LSTM model.

3. **Model Definition:**
   - The model consists of an Embedding layer (initialized with Word2Vec vectors), a Bidirectional LSTM layer, and fully connected dense layers with dropout for regularization.

4. **Model Training:**
   - The model is trained using the RMSprop optimizer with binary cross-entropy loss. Early stopping and learning rate reduction on plateau are used as callbacks to improve training efficiency.

5. **Model Evaluation:**
   - The model is evaluated on the validation set, and the AUC-ROC score is calculated. The model's predictions are made on the test set, and results are saved for submission.

6. **Model Summary:**
   - The summary of the model architecture is printed to provide an overview of the layers and parameters.

7. **Prediction and Submission:**
   - Predictions are made on the test data, and the results are saved in a `submission.csv` file for submission.

## Example Output

- **Model Summary:**

    ```
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 123, 150)          20346000  
                                                                     
     bidirectional (Bidirection  (None, 128)               110080    
     al)                                                              
                                                                     
     dense (Dense)               (None, 10)                1290      
                                                                     
     dropout (Dropout)           (None, 10)                0         
                                                                     
     dense_1 (Dense)             (None, 1)                 11        
                                                                     
    =================================================================
    Total params: 20457381 (78.04 MB)
    Trainable params: 20457381 (78.04 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    ```

- **Training Output:**
  
    ```sh
    Epoch 1/30
    86/86 [==============================] - 93s 960ms/step - loss: 0.4801 - acc: 0.7712 - val_loss: 0.4027 - val_acc: 0.8378 - lr: 0.0010
    ...
    Epoch 30/30
    86/86 [==============================] - 72s 832ms/step - loss: 0.1179 - acc: 0.9556 - val_loss: 0.2519 - val_acc: 0.9239 - lr: 0.0010
    ```

- **AUC-ROC Score:**

    ```sh
    roc_auc_score: 0.9763632451508162
    ```

- **Submission File:**
  - The predictions are saved in `submission.csv` for submission.
