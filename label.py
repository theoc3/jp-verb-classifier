import pandas as pd
import os.path
from classify import classify_verb, find_verb
from tqdm import tqdm

def extract_excel(raw_path,excel_path):
    if not os.path.exists(raw_path):
        print("CSV file not found. Converting Excel to CSV.")
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
        df.to_csv(raw_path, index=False)
    else:
        print("CSV file already exists. Skipping Excel conversion.")

def extract_utf(raw_path, utf_path):
    # Check if the output file already exists
    if not os.path.exists(raw_path):
        print(f"Raw sentences file not found. Processing UTF file: {utf_path}")
        
        # Read the file as text
        with open(utf_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Extract sentences from A: lines
        sentences = []
        ids = []
        translations = []
        
        for line in lines:
            if line.startswith('A:'):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    japanese_text = parts[0].replace('A:', '').strip()
                    
                    # Extract ID if available
                    id_match = None
                    if '#ID=' in parts[1]:
                        english_translation, id_part = parts[1].split('#ID=')
                        id_match = id_part.strip()
                    else:
                        english_translation = parts[1].strip()
                    
                    sentences.append(japanese_text)
                    translations.append(english_translation)
                    ids.append(id_match)
        
        sentences_df = pd.DataFrame({
            '#日本語(原文)': sentences,
            'English': translations,
            'ID': ids
        })
        
        # Save the raw sentences
        sentences_df.to_csv(raw_path, index=False, encoding='utf-8')
    else:
        print("CSV file already exists. Skipping UTF Conversion")

def label(raw_path,output_path):
    sentences_df = pd.read_csv(raw_path)

    print(sentences_df.head())

    def find_and_classify(text):
        try:
            verbs = find_verb(text)
            result = []
            for i in verbs:
                verb_info = classify_verb(i)
                if verb_info is not None:
                    result.append(verb_info)
            return result
        except Exception as e:
            print(f"Skipping sentence due to error: {e}, text: {text}")
            return []

    tqdm.pandas(desc="Processing sentences")
    sentences_df['verb_data'] = sentences_df['#日本語(原文)'].progress_apply(find_and_classify)

    # Find the maximum number of verbs in any sentence
    max_verbs = sentences_df['verb_data'].apply(len).max()

    # Create separate columns for each verb and verb type
    for i in range(max_verbs):
        sentences_df[f'verb{i+1}'] = sentences_df['verb_data'].apply(
            lambda x: x[i][0] if x is not None and i < len(x) else None
        )
        sentences_df[f'verbType{i+1}'] = sentences_df['verb_data'].apply(
            lambda x: x[i][1] if x is not None and i < len(x) else None
        )

    # Remove the intermediate verb_data column if desired
    sentences_df = sentences_df.drop('verb_data', axis=1)

    # Save the updated dataframe
    sentences_df.to_csv(output_path, index=False)



# extract_excel('data/data_1/sentences_raw.csv',
#               'data/data_1/T15-2020.1.7.xlsx',)

# label('data/data_1/sentences_raw.csv',
#       'data/data_1/sentences_labeled.csv')

extract_utf('data/data_2/sentences_raw.csv',
              'data/data_2/examples.utf')

label('data/data_2/sentences_raw.csv',
      'data/data_2/sentences_labeled.csv')