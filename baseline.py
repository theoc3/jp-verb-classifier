import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pickle
import os

df = pd.read_csv('data/data_2/sentences_with_segments.csv',low_memory=False)
print(f"Loaded {len(df)} processed sentences with individual segments")

model_path = 'models/baseline_model.pkl'

data = []
for idx, row in df.iterrows():
    for i in range(1, 11):  # There are up to 10 verbs per sentence
        verb_col = f'verb{i}'
        type_col = f'verbType{i}'
        segment_col = f'segment{i}'
        
        if pd.notna(row[verb_col]) and pd.notna(row[type_col]) and pd.notna(row[segment_col]):
            data.append({
                'segment': row[segment_col],
                'verb': row[verb_col],
                'verb_type': row[type_col]
            })

verb_df = pd.DataFrame(data)
print(f"Extracted {len(verb_df)} verb-segment pairs")

if os.path.exists(model_path):
    print(f"Loading existing model from '{model_path}'")
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
else:
    # remove classes that have only 1 member
    class_counts = verb_df['verb_type'].value_counts()
    single_classes = class_counts[class_counts == 1].index
    print(f"Removing {len(single_classes)} classes with only 1 example: {', '.join(single_classes)}")
    verb_df = verb_df[~verb_df['verb_type'].isin(single_classes)]
    print(f"Remaining examples: {len(verb_df)}")

    # character n-grams (1-3)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform(verb_df['segment'])
    y = verb_df['verb_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Model saved to '{model_path}'")

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    
    report_path = model_path.replace('.pkl', '_evaluation.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Evaluation report saved to '{report_path}'")

##########################################################################################

def predict_verb_type(segment):
    segment_vec = vectorizer.transform([segment])
    prediction = model.predict(segment_vec)[0]
    probabilities = model.predict_proba(segment_vec)[0]
    confidence = max(probabilities)
    return prediction, confidence

### TEST PREDICTION
test_segments = ["誰が一番に着く", "か私には分かりません"]
for segment in test_segments:
    pred_type, confidence = predict_verb_type(segment)
    print(f"Segment: {segment}")
    print(f"Predicted verb type: {pred_type} (confidence: {confidence:.2f})")
