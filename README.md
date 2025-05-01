# jp-verb-classifier

## Description
 A Bi-directional Long Short-Term Memory (BiLSTM) Classifer for Japanese sentences that categorizes the verbs contained within by their tense/voice e.g. passive, past, participle, etc.

```text
jp-verb-classifier/
├── data/                                   *data files*
│   ├── data_1/                              *data files for the SNOW T15 Corpus (not used in final model)*
│   ├── ├── sentences_cleaned.csv             *Step 3. cleaned sentences for T15*
│   ├── ├── sentences_labeled.csv             *Step 2. labeled sentences for T15*
│   ├── ├── sentences_raw.csv                 *Step 1. raw sentences for T15*
│   ├── ├── sentences_with_segments.csv       *Step 4. sentences with segments extracted by verb for T15*
│   │   └── T15-2020.1.7.xlsx                 *original dataset*
│   └── data_2/                             *data files for the Tanaka Corpus*
│       ├── examples.utf                      *original dataset*
│       ├── sentences_cleaned.csv             *Step 3. cleaned sentences for Tanaka*
│       ├── sentences_labeled.csv             *Step 2. labeled sentences for Tanaka*
│       ├── sentences_raw.csv                 *Step 1. raw sentences for Tanaka*
│       └── sentences_with_segments.csv       *Step 4. sentences with segments extracted by verb for Tanaka*
├── legacy/                                  *legacy files from previous attempts at labeling pipeline, no longer works*
│   ├── baseline.py                          
│   ├── inflections.py                     
│   ├── tense_check_test.py
│   └── tense_check.py
├── models/                                  *saved models and related classifier evaluation data*
│   ├── baseline_model_evaluation.txt         *evaluation of baseline*
│   ├── baseline_model.pkl                    *saved baseline model (not included in repo due to size)* 
│   ├── bilstm_model_evaluation.txt           *evaluation of bilstm*
│   └── bilstm_model.pt                       *saved bilstm model*
├── plots/                                   *figures
│   ├── baseline_metrics.png                  *bar chart of precision, recall, and F1 for baseline*
│   ├── bilstm_metrics.png                    *bar chart of precision, recall, and F1 for bilstm*
│   └── label-process-graph.png               *visual flow-chart of how verbs are classified by classify.py*
├── baseline.py                              *trains or loads new baseline model* 
├── bilstm_epochs.csv                        *epoch training metrics from training bilstm*
├── classify.py                              *module for classifying verbs* 
├── conj_tags_to_inflections.csv             *conjugation rule tagset for spacy tags --> mecab tags*
├── conjugation_tags.csv                     *conjugation rule tagset for mecab tags --> tags recognized by inflections.py*
├── data_setup.py                            *Completes step 3 and 4 of data processing*
├── inflection_types.csv                     *list of root conjugation rules*
├── inflections.py                           *conjugation/inflection script that takes a root verb and its conjugation rule, returning a full dictionary of conjugations*
├── label.py                                 *Completes step 1 and 2 of data processing*
├── lstm.py                                  *trains or loads new lstm model*
├── report.py                                *generates tables and figures for metrics foudn in /plot*
└── sentence_tokenize.py                     *depreceated, but saves a .pkl of sentence_cleaned.csv with an additional column containing spacy token data*
```

## Installation

Only ```python<=3.12.3``` will work as of now, due to [compatibility issues](https://github.com/explosion/spaCy/issues/13550) with spaCy. 

### Corpora

The corpora are included in the repository, but can be re-downloaded: [Tanaka Corpus](http://edrdg.org/wiki/index.php/Tanaka_Corpus) (used in final model) [SNOW T15](https://www.jnlp.org/GengoHouse/snow/t15)

The SNOW T15 corpus was dropped due to its low number of examples for conjugations like passive and potential (which is the focus of this project) 

### spaCy

For spaCy, follow [these instructions](https://spacy.io/models/ja) and the model used is ```ja_core_news_trf```

These should be the commands:
```
pip install spacy
python -m spacy download ja_core_news_trf
```

### MeCab

For MeCab, the [official installation instructions](https://taku910.github.io/mecab/#download) are in Japanese, but the relevant commands are:

MeCab itself:
```
tar zxfv mecab-0.996.tar.gz
cd mecab-0.996
./configure 
make
make check
su
make install
```

MeCab's Dictionary:
```
% tar zxfv mecab-ipadic-2.7.0-20070610.tar.gz
% mecab-ipadic-2.7.0-20070610
% ./configure
% make
% su
# make install
```

[Here](https://drive.google.com/drive/folders/1-xcDvik4983YEEAOfkWhliDmarfqani_) is a list of dictionaries to use for MeCab in case there are issues with the above commands. 

This [repository](https://github.com/buruzaemon/natto-py?tab=readme-ov-file) also requires mecab to work, so if there are issues with finding the dictionary's path refer to this. 

### Other Libraries

```pip install pandas numpy matplotlib scikit-learn torch pickle jamdict jamdict-data```

## Usage

### test.csv
To manually use examples for each model to attempt to classify, modify test.csv where each row is a sentence **segment**, a sentence containing only a single verb. If a sentence with multiple verbs is used, the first verb in the sentence is classified when using the baseline, and the last verb is used in the bilstm. 

### baseline.py

Run ```python3 baseline.py```, where a logistic regression model will either be loaded or trained on ```/data/data_2/sentences_with_segments.csv```. If a new model is trained, new metrics are saved to baseline_model_evaluation.txt. Then, examples from test.csv will be classified and printed. 

#### Examples:
```text
Example Predictions:
Segment: 誰が一番に着く
Predicted verb type: dict (confidence: 0.90)
Segment: か私には分かりません
Predicted verb type: negative-polite (confidence: 1.00)
Segment: 食べられる
Predicted verb type: passive (confidence: 0.98)
Segment: お姉さんに私のりんごを食べられた
Predicted verb type: passive-past (confidence: 0.96)
Segment: お姉さんは私のりんごが食べられた
Predicted verb type: potential-past (confidence: 0.80)
Segment: お姉さんに私のりんごを食べられました
Predicted verb type: passive-past-polite (confidence: 0.99)
Segment: お姉さんは私のりんごが食べられました
Predicted verb type: passive-past-polite (confidence: 0.97)
Segment: お姉さんに私のりんごを食べられて
Predicted verb type: passive-participle (confidence: 0.97)
Segment: お姉さんは私のりんごが食べられて
Predicted verb type: passive-participle (confidence: 0.90)
Segment: 私に食べられた
Predicted verb type: passive-past (confidence: 0.97)
Segment: 私は食べられた
Predicted verb type: passive-past (confidence: 0.97)
Segment: した私は食べる
Predicted verb type: past (confidence: 0.86)
```

### lstm.py

Run ```python3 lstm.py```, where a bidirectional long short-term memory model will either be loaded or trained on ```/data/data_2/sentences_with_segments.csv```. If a new model is trained, new metrics are saved to bilstm_model_evaluation.txt. Then, examples from test.csv will be classified and printed. 

#### Examples:
```text
Segment: 誰が一番に着く
Predicted verb type: dict (confidence: 1.00)
Segment: か私には分かりません
Predicted verb type: negative-polite (confidence: 1.00)
Segment: 食べられる
Predicted verb type: passive (confidence: 1.00)
Segment: お姉さんに私のりんごを食べられた
Predicted verb type: passive-past (confidence: 1.00)
Segment: お姉さんは私のりんごが食べられた
Predicted verb type: potential-past (confidence: 1.00)
Segment: お姉さんに私のりんごを食べられました
Predicted verb type: passive-past-polite (confidence: 1.00)
Segment: お姉さんは私のりんごが食べられました
Predicted verb type: potential-past-polite (confidence: 0.98)
Segment: お姉さんに私のりんごを食べられて
Predicted verb type: passive-participle (confidence: 1.00)
Segment: お姉さんは私のりんごが食べられて
Predicted verb type: potential-participle (confidence: 1.00)
Segment: 私に食べられた
Predicted verb type: passive-past (confidence: 1.00)
Segment: 私は食べられた
Predicted verb type: passive-past (confidence: 1.00)
Segment: した私は食べる
Predicted verb type: dict (confidence: 1.00)
```

## References & Libraries 

- [Python 3 Reference Manual](https://www.python.org/) Python v3.12.3 (Highest version compatible with spaCy)
- [Tanaka Corpus](http://edrdg.org/wiki/index.php/Tanaka_Corpus) Managed by the Tatoeba Corpus project, UTF-8 downloaded
- [SNOW T15 Corpus](https://www.jnlp.org/GengoHouse/snow/t15) Japanese Simplified Corpus with Core Vocabulary (not used in final model)
- [spaCy: Industrial-strength Natural Language Processing in Python](https://spacy.io) Tokenization, lemmatization, POS-tagging, conjugation rule type
- [spaCy (Japanese models)](https://spacy.io/models/ja) Japanese models
- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/) Training/test split, logistic regression, classifier evaluation
- [PyTorch](https://pytorch.org) Dataset, dataloader, BiLSTM Model
- [pandas: powerful Python data analysis toolkit](https://pandas.pydata.org) Data cleaning
- [NumPy (Nature article)](https://www.nature.com/articles/s41586-020-2649-2) Data cleaning
- [pickle — Python object serialization](https://docs.python.org/3/library/pickle.html) Model saving and loading
- [Matplotlib](https://ieeexplore.ieee.org/document/4160265) Plot graphing
- [Jamdict: A Japanese-Multilingual Dictionary Interface](https://jamdict.readthedocs.io/en/latest/) Lemmatization verification
- [Japanese Verb Conjugation Scripts](https://github.com/jmdict-kindle/jmdict-kindle) Original script for classifying verbs based on rules
- [Japanese MeCab Part-of-Speech Tagset](https://www.sketchengine.eu/tagset-jp-mecab/) Reference for classes of Japanese verbs
- [MeCab: Yet Another Part-of-Speech and Morphological Analyzer](https://taku910.github.io/mecab/) Used by spaCy

 
