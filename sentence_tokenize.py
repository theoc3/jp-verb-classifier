import spacy
nlp = spacy.load("ja_core_news_trf")

import pandas as pd
from tqdm import tqdm

csv_path = 'data/sentences_cleaned.csv'

df = pd.read_csv(csv_path)
#df = df.head(5)
tqdm.pandas(desc="tokenizing sentences")
df['tokens'] = df['sentence'].progress_apply(nlp)

df.to_pickle('data/sentences_processed.pkl')

loaded = pd.read_pickle('data/sentences_processed.pkl')
def extract_token_info(doc):
    return [{'text': token.text, 'lemma': token.lemma_, 
             'pos': token.pos_, 'dep': token.dep_} 
            for token in doc]
    
#loaded['token_features'] = loaded['tokens'].apply(extract_token_info)
#loaded.to_csv('processed_sentences.csv', index=False)
print(loaded)

