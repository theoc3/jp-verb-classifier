import pandas as pd
import re

def data_clean(input_path,output_path):

    df = pd.read_csv(input_path,low_memory=False)

    df = df.drop(['#やさしい日本語', '#英語(原文)'], axis=1, errors='ignore')

    df = df.rename(columns={'#日本語(原文)': 'sentence'})

    df_processed = df.dropna(subset=['verb1'])

    df_processed.to_csv(output_path, index=False)
    
def data_setup(input_path,output_path):
    df = pd.read_csv(input_path,low_memory=False)

    df['segment1'] = None
    df['segment2'] = None
    df['segment3'] = None
    df['segment4'] = None
    df['segment5'] = None
    df['segment6'] = None
    df['segment7'] = None
    df['segment8'] = None
    df['segment9'] = None
    df['segment10'] = None

    for index, row in df.iterrows():
        sentence = row['sentence']
    
        verbs = []
        for i in range(1, 11): # 10 possible verbs
            verb_key = f'verb{i}'
            if pd.notna(row[verb_key]) and row[verb_key]:
                verbs.append(row[verb_key])
        
        if not verbs:
            continue
            
        # positions of verb in sentence
        positions = []
        remaining_sentence = sentence
        
        for verb in verbs:
            if not verb or pd.isna(verb):
                continue
            
            pos = remaining_sentence.find(verb)
            if pos >= 0:
                if len(positions) == 0:
                    positions.append(pos + len(verb))
                else:
                    positions.append(positions[-1] + pos + len(verb))
                
                remaining_sentence = sentence[positions[-1]:]
    
        # segment accumulator
        segments = []
        start = 0
        
        for pos in positions:
            segments.append(sentence[start:pos])
            start = pos
        
        for i, segment in enumerate(segments, 1):
            if i <= 4: # only 4 segments
                df.at[index, f'segment{i}'] = segment

    df.to_csv(output_path, index=False)
    
# data_clean('data/data_1/sentences_labeled.csv','data/data_1/sentences_cleaned.csv')
# data_setup('data/data_1/sentences_cleaned.csv','data/data_1/sentences_with_segments.csv')


data_clean('data/data_2/sentences_labeled.csv','data/data_2/sentences_cleaned.csv')
data_setup('data/data_2/sentences_cleaned.csv','data/data_2/sentences_with_segments.csv')