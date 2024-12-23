

# COVID-19 Tweet Clustering using GSDMM

This notebook performs clustering of COVID-19-related tweets using the **GSDMM (Gibbs Sampling for Dirichlet Multinomial Mixture Model)** algorithm. The main steps involve text preprocessing, tokenization, lemmatization, and clustering of tweets into groups based on similar content. Below is a summary of the approach and methodology used:

### Methodology:
1. **Data Collection**: 
   - The dataset contains tweets related to COVID-19, with a focus on sentiment analysis.
   
2. **Text Preprocessing**: 
   - Tokenization: The tweets are split into individual words.
   - Removal of stopwords: Common words (e.g., "the", "and", etc.) are removed to focus on meaningful words.
   - Lemmatization: Words are reduced to their base form (e.g., "running" to "run").
   - N-grams Creation: Bigram and trigram models are generated to capture word relationships.

3. **Modeling**: 
   - GSDMM is applied to the preprocessed tweets, where different combinations of hyperparameters (Alpha and Beta) are tested to optimize clustering.
   - The optimal number of clusters (K) is set to 4 for this analysis.

4. **Clustering and Evaluation**:
   - Each tweet is assigned to the most probable cluster based on the word distribution learned by the model.
   - The results are evaluated based on the loss function, which measures how well the documents are distributed across clusters.

5. **Results**: 
   - The final output includes the cluster assignments for each tweet, along with the most frequent words in each cluster.
   - Results are saved into a CSV file named `ClusteredTweets.csv`.

### Key Functions:
- `sent_to_words()`: Tokenizes the tweet texts into words.
- `make_n_grams()`: Generates bigrams and trigrams to enhance the model's understanding of word relationships.
- `remove_stopwords()`: Removes common stopwords to focus on meaningful words.
- `lemmatization()`: Reduces words to their base form using the Spacy NLP library.
- `choose_best_label()`: Assigns each tweet to the most likely cluster based on the word distribution in each cluster.

### Requirements:
- **Python 3.x**
- **Libraries**:
  - `numpy`
  - `pandas`
  - `gsdmm`
  - `gensim`
  - `spacy`
  - `nltk`

### How to Run:
1. Install the required libraries:
   ```bash
   pip install numpy pandas gsdmm gensim spacy nltk
   ```
2. Download the Spacy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. Run the notebook and observe the results. The final output CSV file will be saved as `ClusteredTweets.csv`.

