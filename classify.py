import spacy
nlp = spacy.load("ja_core_news_trf")

from jamdict import Jamdict
jam = Jamdict()

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
    add_aux = False
    
    ni = False
    wo = False
    ga = False
    
    last_particles = []
    full_verb = []
    index = 0
    
    # Track potential noun part of する compound verb
    potential_noun = None
    
    for i in nlp_sentence:
        #print(i.pos_,i.text,i.lemma_,i.morph)
        if i.pos_ == 'ADP':
            if i.text == 'を':
                last_particles.append(('wo', index))
            elif i.text == 'が':
                last_particles.append(('ga', index))
            elif i.text == 'に':
                ni = True
        if i.pos_ == 'VERB': 
            current_verb = i
            add_aux = True
            
            # Check for suru verb
            if all('\u4E00' <= char <= '\u9FFF' for char in i.text):
                if len(i.text) == 2 and i.text != '出来':
                    continue
                else: 
                    full_verb.append((index,i))
            else:
                full_verb.append((index,i))
 
            # Check if particles preceded this verb
            for particle, p_idx in last_particles:
                if index - p_idx == 1:  # Check if particle is next to verb
                    if particle == 'wo':
                        wo = True
                    elif particle == 'ga':
                        ga = True
            last_particles = []
            
        elif i.pos_ == 'AUX' and add_aux:
            full_verb.append((index, i))
        # elif i.pos_ == 'AUX' # Future desu case
        
        elif i.pos_ == 'SCONJ' and add_aux:
            full_verb.append((index, i))
            all_verbs.append((full_verb,ni,wo,ga))
            add_aux = False
            full_verb = []
            ni = False
            wo = False
            ga = False
        elif i.pos_ != 'AUX' and add_aux:
            all_verbs.append((full_verb,ni,wo,ga))
            add_aux = False
            full_verb = []
            ni = False
            wo = False
            ga = False
        index += 1
    
    # Append the last full_verb if the sentence ends with a verb or auxiliary verb
    if add_aux and full_verb:
        all_verbs.append((full_verb,ni,wo,ga))
    
    result = []
    for verb in all_verbs:
        full_verb_form = ''.join([token.text for _, token in verb[0]])
        result.append((full_verb_form, verb[0],verb[1:]))
    return result

# input is individual entry from find_verb
def get_verb_type(verb):
    token = verb[1][0][1]
    morph = token.morph.to_dict()
    lemma = verb[1][0][1].lemma_
    if lemma == "行く":
        return "v5k-s"
    if lemma == "ある":
        return "v5r-i"
    if 'Inflection' not in morph:
        raise ValueError("Missing Inflection in morphology data")
    elif "文語" in morph['Inflection']:
        raise ValueError("Classical Verb")
    elif "形容詞" in morph['Inflection']:
        raise ValueError("Desu Verb")
    else:
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
    
    
    
    ni = verb[2][0]
    wo = verb[2][1]
    ga = verb[2][2]
    
    exceptionDict = {
    'え':'う',
    'け':'く',
    'せ':'す',
    'て':'つ',
    'ね':'ぬ',
    'へ':'ふ',
    'め':'む',
    'げ':'ぐ',
    'ぜ':'ず',
    'で':'づ',
    'べ':'ぶ',
    'ぺ':'ぷ'
    }
    
    for key,value in conjugations.items():
        if full_verb_form == value:
            if 'potential' in key:
                if ni == True:
                    if wo == True:
                        return full_verb_form, key.replace('potential', 'passive')
                    else:
                        if ga == True:
                            return full_verb_form, key
                        else: 
                            return full_verb_form, key.replace('potential', 'passive')
                else: 
                    if wo == True:
                        return full_verb_form, key.replace('potential', 'passive')
                    else:
                        if ga == True:
                            return full_verb_form, key
                        else:
                            return full_verb_form, key.replace('potential', 'passive')
            elif verb[1][0][1].lemma_[-2] in exceptionDict.keys() and jam.lookup(verb[1][0][1].lemma_) == []:
                if full_verb_form[:2] == "見え" or full_verb_form[:3] == "聞こえ":
                    return full_verb_form, key
                else:
                    if key == 'dict':
                        return full_verb_form, "potential"
                    else:
                        return full_verb_form, "potential-" + key
            else:
                return full_verb_form, key
    return None
    
    

# find_verb output format: List of 2-Tuples representing verbs found: (full_verb,[(index,token)])
# token object: token.text,token.lemma_,token.pos_,token.morph,token.is_stop,token.tag_

# v1 
# ichidan_verb = find_verb("""
#                 食べる、
#                 食べ、
#                 食べた、
#                     食べたら、
                    
#                 食べない、
#                     食べなかった、
#                         食べなかったら、
#                     食べなく、
#                     食べないで、
                    
#                 食べられる、
#                     食べられた、
#                         食べられたら、
#                     食べられない、
#                         食べられなかった、
#                             食べられなかったら、
#                         食べられなく、
#                         食べられないで、
#                     食べられれば、##
#                     食べられなければ、##
#                     食べられなきゃ、##
#                     食べられます、
#                     食べられました、
#                         食べられましたら、
#                     食べられません、
#                     食べられましょう、
                
#                 食べれる、
#                     食べれた、
#                         食べれたら、
#                     食べれない、
#                         食べれなかった、
#                             食べれなかったら、
#                         食べれなく、
#                         食べれないで、
#                     食べれれば、##
#                     食べれなければ、##
#                     食べれなきゃ、##
#                     食べれます、
#                     食べれました、
#                         食べれましたら、
#                     食べれません、
#                     食べれましょう、
                
#                 食べられる、
#                     食べられた、
#                         食べられたら、
#                     食べられない、
#                         食べられなかった、
#                             食べられなかったら、
#                         食べられなく、
#                         食べられないで、
#                     食べられれば、##
#                     食べられなければ、##
#                     食べられなきゃ、##
#                     食べられます、
#                     食べられました、
#                         食べられましたら、
#                     食べられません、
#                     食べられましょう、
                        
#                 食べさせる、
#                     食べさせた、
#                         食べさせたら、
#                     食べさせない、
#                         食べさせなかった、
#                             食べさせなかったら、
#                         食べさせなく、
#                         食べさせないで、
#                     食べさせば、
#                     食べさせなければ、
#                     食べさせなきゃ、##
#                     食べさせます、
#                     食べさせました、
#                         食べさせましたら、
#                     食べさせません、
#                     食べさせましょう、
                            
#                 食べさせられる、
#                     食べさせられた、
#                         食べさせられたら、
#                     食べさせられない、
#                         食べさせなかった、
#                             食べさせなかったら、
#                         食べさせなく、
#                         食べさせないで、
#                     食べさせられれば、##
#                     食べさせられなければ、##
#                     食べさせられなきゃ、##
#                     食べさせられます、
#                     食べさせられました、   
#                         食べさせられましたら、
#                     食べさせられません、
#                     食べさせられましょう、
                
#                 食べれば、
#                 食べれなければ、
#                     食べなきゃ、
                
#                 食べろ、
#                 食べよう、
#                 食べたい、
#                 食べたかった、
#                     食べたかったら、
#                 食べたく、
#                 食べます、
#                 食べました、
#                     食べましたら、
#                 食べません、
#                 食べましょう、
#                 """)

# godan_verb_k = find_verb("""
#                 行く、
#                 行き、
#                 行った、
#                     行ったら、
                    
#                 行かない、
#                     行かなかった、
#                         行かなかったら、
#                     行かなく、
#                     行かないで、
                    
#                 行ける、
#                     行けた、
#                         行けたら、
#                     行けない、
#                         行けなかった、
#                             行けなかったら、、
#                         行けなく、
#                         行けないで、
#                     行けば、##
#                     行けなければ、##
#                     行けなきゃ、##
#                     行けます、
#                     行けました、
#                         行けましたら、
#                     行けません、
#                     行けましょう、
                
#                 行かれる、
#                     行かれた、
#                         行かれたら、
#                     行かれない、
#                         行かれなかった、
#                             行かれなかったら、
#                         行かれなく、
#                         行かれないで、
#                     行かれれば、##
#                     行かれなければ、##
#                     行かれなきゃ、##
#                     行かれます、
#                     行かれました、
#                         行かれましたら、
#                     行かれません、
#                     行かれましょう、
                        
#                 行かせる、
#                     行かせた、
#                         行かせたら、
#                     行かせない、
#                         行かせなかった、
#                             行かせなかったら、
#                         行かせなく、
#                         行かせないで、
#                     行かせれば、
#                     行かせなければ、
#                     行かせなきゃ、##
#                     行かせます、
#                     行かせました、
#                         行かせましたら、
#                     行かせません、
#                     行かせましょう、
                            
#                 行かせられる、
#                     行かせられた、
#                         行かせられたら、
#                     行かせられない、
#                         行かせられなかった、
#                             行かせられなかったら、
#                         行かせられなく、
#                         行かせられないで、
#                     行かせられれば、##
#                     行かせられなければ、##
#                     行かせられなきゃ、##
#                     行かせられます、
#                     行かせられました、   
#                         行かせられましたら、
#                     行かせられません、
#                    行かせられましょう、
                
#                 行けば、
#                 行けなければ、
#                     行けなきゃ、
                
#                 行けろ、
#                 行こう、
#                 行きたい、
#                 行きたかった、
#                     行きたかったら、
#                 行きたく、
#                 行きます、
#                 行きました、
#                     行きましたら、
#                 行きません、
#                 行きましょう、
#                 """)

# copula = find_verb("""
#                 です、
#                     でした、
#                         でしたら、
                
#                 ではない、じゃない、
#                     ではなかった、じゃなかった、
#                         ではなかったら、じゃなかったら、
#                     ではなく、じゃなく、
#                     ではなくて、じゃなくて、
                
#                 であれば、
#                 でなければ、
#                     でなきゃ、
                
#                 でしょう、
#                 だろう、
                
#                 であった、
#                     であったら、
#                 であり、
#                 であって、
                
#                 ではあります、じゃあります、
#                 ではありません、じゃありません、
#                     ではありませんでした、じゃありませんでした、
#                         ではありませんでしたら、じゃありませんでしたら、
                
#                 でございます、
#                     でございました、
#                         でございましたら、
#                     でございません、
#                         でございませんでした、
#                             でございませんでしたら、
                
#                 だった、
#                     だったら、
#                 だって、
                
#                 ですが、
#                 ですけど、
#                 ですから、
#                 """)

# verbs = find_verb("""
# 「風と共に去りぬ」を読む。""")

# print(verbs)
# for i in verbs:

#     print(classify_verb(i))
    #pprint.pprint(get_conjugations(i))
    
    
# from jamdict import Jamdict

# jam = Jamdict()
# result = jam.lookup("行ける")

# for entry in result.entries:
#     print(entry)
    





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


