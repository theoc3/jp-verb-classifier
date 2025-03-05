import spacy
nlp = spacy.load("ja_core_news_sm")

import csv

import inflections

import pprint

def generate_tagset(filepath):
    tagset = {}
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                key = row[0]
                value = row[1]
                tagset[key] = value
    return tagset

# https://www.sketchengine.eu/tagset-jp-mecab/
mecab_tagset = generate_tagset("./conjugation_tags.csv")
inflection_tagset = generate_tagset("./conj_tags_to_inflections.csv")

def find_verb(sentence):
    all_verbs = []
    nlp_sentence = nlp(sentence)
    # for token in nlp_sentence:
    #     pprint.pprint((token.text,token.lemma_,token.pos_,token.morph,token.is_stop,token.tag_))
    add_aux = False
    full_verb = []
    index = 0
    for i in nlp_sentence:
        if i.pos_ == 'VERB': 
            current_verb = i
            add_aux = True
            full_verb = [(index, i)]
        elif i.pos_ == 'AUX' and add_aux:
            full_verb.append((index, i))
        elif i.pos_ == 'SCONJ' and add_aux:
            full_verb.append((index, i))
            all_verbs.append(full_verb)
            add_aux = False
            full_verb = []
        elif i.pos_ != 'AUX' and add_aux:
            all_verbs.append(full_verb)
            add_aux = False
            full_verb = []
        index += 1
    
    # Append the last full_verb if the sentence ends with a verb or auxiliary verb
    if add_aux and full_verb:
        all_verbs.append(full_verb)

    # for verb in all_verbs:
    #     full_verb_form = ''.join(["する" if i.form == "為る" else i.form for i in verb])
    #     if verb[0].lemma == "する" or verb[0].lemma == "為る":
    #         result.append(("する",full_verb_form,verb))
    #     elif verb[1].lemma == "為る":
    #         result.append(("する",full_verb_form[2:],verb))
    #     else:
    #         result.append((verb[0].lemma,full_verb_form,verb))
    result = []
    for verb in all_verbs:
        full_verb_form = ''.join([token.text for _, token in verb])
        result.append((full_verb_form, verb))
    return result

# input is individual entry from find_verb
def get_verb_type(verb):
    token = verb[1][0][1]
    morph = token.morph.to_dict()
    rule = morph['Inflection'].split(';')[0]
    return inflection_tagset[mecab_tagset[rule]]
    
# input is individual entry from find_verb
def get_conjugations(verb):
    token = verb[1][0][1]
    verb_type = get_verb_type(verb)
    return inflections.inflect(token.lemma_,verb_type)


def full_inflections(verb):
    conjugations = get_conjugations(verb)
    if "potential" in conjugations:
        potential_conj = inflections.inflect(conjugations["potential"], "v1")
        for key, value in potential_conj.items():
            conjugations[f"potential-{key}"] = value
    
    if "potential-colloquial" in conjugations:
        potential_colloquial_conj = inflections.inflect(conjugations["potential-colloquial"], "v1")
        for key, value in potential_colloquial_conj.items():
            conjugations[f"potential-colloquial-{key}"] = value
    
    if "passive" in conjugations:
        passive_conj = inflections.inflect(conjugations["passive"], "v1")
        for key, value in passive_conj.items():
            conjugations[f"passive-{key}"] = value
    
    if "causative" in conjugations:
        causative_conj = inflections.inflect(conjugations["causative"], "v1")
        for key, value in causative_conj.items():
            conjugations[f"causative-{key}"] = value
    
    if "causative-passive-colloquial" in conjugations:
        causative_passive_colloquial_conj = inflections.inflect(conjugations["causative-passive-colloquial"], "v1")
        for key, value in causative_passive_colloquial_conj.items():
            conjugations[f"causative-passive-colloquial-{key}"] = value
    
    if "causative-passive" in conjugations:
        causative_passive_conj = inflections.inflect(conjugations["causative-passive"], "v1")
        for key, value in causative_passive_conj.items():
            conjugations[f"causative-passive-{key}"] = value
    
    return conjugations

# input is individual entry from find_verb
def classify_verb(verb):
    full_verb_form = verb[0]
    conjugations = full_inflections(verb)
    
    for key,value in conjugations.items():
        if full_verb_form == value:
            return full_verb_form, key
    return None
    
    

# find_verb output format: List of 2-Tuples representing verbs found: (full_verb,[(index,token)])
# token object: token.text,token.lemma_,token.pos_,token.morph,token.is_stop,token.tag_
verbs = find_verb("聞いてほしい、食べられます、食べられる")
for i in verbs:
    print(classify_verb(i))
    #pprint.pprint(get_conjugations(i))
    





# doc = nlp("歌わせられる、歌わされる、妹は母に皿を洗わせられる")

# for token in doc:
#     print(token.text,token.lemma_,token.pos_,token.morph,token.is_stop,token.tag_)
#     print(type(token))
#     print(type(token.text))
#     print(token.morph.to_dict())
    
# nominal 食べ
# past　食べた
# negative 食べない
# participle　食べて
# potential　食べられる
# potential-colloquial 食べれる ###
# passive　食べられる
# causative　食べさせる
# provisional-conditional　食べれば
# imperative　食べろ
# volitional　食べよう

# negative-nominal　食べなく
# negative-past　食べなかった
# negative-participle　食べなくて
# negative-provisional-conditional　食べなければ
# negative-provisional-conditional-colloquial　食べなきゃ
# conditional　食べたら
# wish　食べたい
# wish-past　食べたかった
# wish-nominal　食べたく
# polite　食べます
# past-polite　食べました
# negative-polite　食べません
# volitional-polite　食べましょう



# causative-passive-colloqial 呼ばされる
#   causative-passive-colloqial + nominal etc
#   causative-passive + nominal etc
#   potential + nominal etc
#   potential-colloqial + nominal etc
#   passive + nominal etc
#   causative + nominal etc
# + nominal
# + past
# + negative
# + participle
# + potential ### 
# + provisional-conditional
# + imperative ###
# + volitional ###
# + negative-nominal
# + negative-past
# + negative-participle
# + negative-provisional-conditional
# + negative-provisional-conditional-colloquial
# + conditional
# + wish
# + wish-past
# + wish-nominal
# + polite
# + past-polite
# + negative-polite
# + negative-past-polite
# + volitional-polite


