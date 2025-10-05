import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm

nltk.download('stopwords', quiet=True, force=False)
nltk.download('wordnet', quiet=True, force=False)

LEMMA = WordNetLemmatizer()
INPUT_CSV_FILE = "cleaned_publications_data.csv"
OUTPUT_CSV_FILE = "publications_with_keywords.csv"

TEXT_COLUMN = "Clean_Text_for_NLP"

TOP_N_KEYWORDS = 10

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
CUSTOM_STOP_WORDS = {'section', 'abstract', 'introduction', 'conclusion', 'result', 'et al', 'fig', 'table', 'doi',
                     'data', 'study', 'research', 'article', 'paper', 'one', 'two', 'three', 'used', 'results',
                     'conclusions', 'found', 'shown','microgravity', 'spaceflight', 'nasa', 'iss', 'station', 'experiment',
                     'astronaut', 'gravity', 'flight', 'specie', 'organism', 'cell'}

ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower()
    # Remove all non-word characters (including numbers, which are often noise)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize and remove stop words
    tokens = []
    for word in text.split():
        if word not in ALL_STOP_WORDS and len(word) > 2:
            lemmatized_word = LEMMA.lemmatize(word)
            tokens.append(lemmatized_word)
    return ' '.join(tokens)

def get_top_tfidf_keywords(tfidf_matrix, feature_names, doc_index, top_n):
    # Get the vector for the specific document
    doc_vector = tfidf_matrix[doc_index]

    dense_vector = doc_vector.toarray().flatten()

    sorted_indices = dense_vector.argsort()[::-1]

    top_n_indices = sorted_indices[:top_n]

    keywords = [feature_names[i] for i in top_n_indices]

    return ", ".join(keywords)

def main_npl_pipeline():
    #load and preprocess data
    try:
        df =pd.read_csv(INPUT_CSV_FILE)
        print(f" Data loaded successfully: {len(df)} publications.")
    except FileNotFoundError:
        print(f" Error: Input CSV file not found at '{INPUT_CSV_FILE}'. Did phase 2 complete successfully?")
        return

    #filter out rows with missing or insufficient text
    initial_count = len(df)
    df.dropna(subset = [TEXT_COLUMN], inplace = True)
    df = df[df[TEXT_COLUMN].str.len() > 50]
    print(f" Removed {initial_count - len(df)} rows with insufficient text for analysis.")

    # Preprocess text data
    print(" Preprocessing text data...")
    df['Processed_Text'] = df[TEXT_COLUMN].apply(preprocess_text)

    #Tfidf Vectorization
    print(" Performing TF-IDF vectorization...")
    tfidf = TfidfVectorizer(min_df = 5, max_df = 0.85)

    #fit model and transform text data
    try:
        tfidf_matrix = tfidf.fit_transform(df['Processed_Text'])
    except ValueError as e:
        print(f" Error during TF-IDF fitting: {e}")
        print(" This means all documents were too short or empty after preprocessing.")
        return
    feature_names = tfidf.get_feature_names_out()

    #Extract Keywords
    print(f" Extracting top {TOP_N_KEYWORDS} keywords for each document...")
    keyword_list = []

    #iterate each document index(row) in the dataframe
    for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords"):
        keywords = get_top_tfidf_keywords(tfidf_matrix, feature_names, i, top_n=TOP_N_KEYWORDS)
        keyword_list.append(keywords)

    #Add keywords to dataframe
    df['Top_Keywords'] = keyword_list

    #Save results
    print("Cleaning up temporary columns and saving final data")

    #check column exists before dropping it
    if 'Preprocessed_Text' in df.columns:
        df.drop(columns=['Preprocessed_Text'], inplace=True)
    else:
        print("Warning: 'Preprocessed_Text' column not found, skipping drop operation.")

    # Save final dataframe
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')

    print(f"\n\n Phase 3 Complete")
    print(f" Final data saved to **{OUTPUT_CSV_FILE}**.")
    print(f" The analysis column is: TfIDF_Keywords_Top_{TOP_N_KEYWORDS}")

if __name__ == "__main__":
    main_npl_pipeline()

#Created and edited by: JaraOps

