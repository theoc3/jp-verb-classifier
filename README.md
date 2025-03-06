# jp-verb-classifier

## Description
 Classify Japanese verbs by their tense/voice i.e. passive, past, participle, etc. 

## Libraries
 [spaCy](https://spacy.io/models/ja) For tokenization, lemmatization, POS-tagging

 [MeCab](https://taku910.github.io/mecab/) tokenization method used by spaCy

 [MeCab tagset](https://www.sketchengine.eu/tagset-jp-mecab/) possible POS tags from MeCab

 [jmdict-kindle](https://github.com/jmdict-kindle/jmdict-kindle/tree/main) conjugation script based off inflections.py found here

## Usage
### Command/Result
```
 python3 classify.py
```

 Currently prints out 4 examples using rule-based script: 
 ```
 ('聞いて', 'participle')
 ('食べられます', 'potential-polite')
 ('食べられる', 'potential')
 ('作りました', 'past-polite')
```
 
## Currently working on...
  Labeling and tokenizing existing corpora to train model
