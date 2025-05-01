# lemmization, POS tagging, and dependency parsing
import unidic2ud
nlp=unidic2ud.load("gendai")

# dictionary lookup
import jamdict
jam = jamdict.Jamdict()

import requests
import json

# conjugation script
import inflections

# kana kanji conversion
import mozcpy
converter = mozcpy.Converter()

# from natto import MeCab
# dicdir = "/opt/homebrew/lib/mecab/dic/ipadic"
# nm = MeCab(f"-d {dicdir}")
# print(nm)
# sysdic = nm.dicts[0]
# print(sysdic)
# MeCab options used:
#
# -F    ... short-form of --node-format
# %m    ... morpheme surface (the original form of the word as it appears in the text)
# %f[0] ... part-of-speech (POS)
# %h ... part-of-speech ID (ipadic)
# %f[1] ... subcategory of POS
# %f[2] ... conjugation type
# %f[3] ... conjugation form
# %f[4] ... base form (lemma)
# %f[5] ... reading
# %f[6] ... pronunciation
#
# -U    ... short-form of --unk-format
#           output ?,?,?,? for morphemes not in dictionary
#
# with MeCab(r'-F%m,%f[0],%h,%f[1],%f[2],%f[3],%f[4],%f[5],%f[6]\n -U?,?,?,?\n') as nm:
#     causative_baseline_test = ["母は弟に野菜を食べさせた。", "先生は生徒に宿題をやらせた。", "父は私に車を運転させた。", "兄は妹に犬の世話をさせた。", "先生は学生に本を読ませた。", "友達は私に歌を歌わせた。", "母は子供に部屋を掃除させた。", "先輩は後輩に荷物を運ばせた。", "店長は店員にドアを開けさせた。", "先生は生徒に作文を書かせしまった。"]   
#     for test in causative_baseline_test:
#         for n in nm.parse(test, as_nodes=True):
#             # only normal nodes, ignore any end-of-sentence and unknown nodes
#             if n.is_nor():
#                 print(n.feature)
#                 print(type(n.feature))
    

def find_verb(sentence):
    all_verbs = []
    nlp_sentence = nlp(sentence)
    add_aux = False
    full_verb = []
    
    for i in nlp_sentence:
        if i.upos == 'VERB': 
            current_verb = i
            add_aux = True
            full_verb = [i]
        elif i.upos == 'AUX' and add_aux:
            full_verb.append(i)
        elif i.upos != 'AUX' and add_aux:
            all_verbs.append(full_verb)
            add_aux = False
            full_verb = []
    
    # Append the last full_verb if the sentence ends with a verb or auxiliary verb
    if add_aux and full_verb:
        all_verbs.append(full_verb)

    result = []
    for verb in all_verbs:
        full_verb_form = ''.join(["する" if i.form == "為る" else i.form for i in verb])
        if verb[0].lemma == "する" or verb[0].lemma == "為る":
            result.append(("する",full_verb_form,verb))
        elif verb[1].lemma == "為る":
            result.append(("する",full_verb_form[2:],verb))
        else:
            result.append((verb[0].lemma,full_verb_form,verb))

    return result
    
# def find_verb(sentence):
#     last_verb = None
#     nlp_sentence = nlp(sentence)
    
    
#     for i in nlp_sentence:
#         if i.upos == 'VERB':
#             last_verb = i
#     add_aux = False
#     full_verb = []
    
#     for i in nlp_sentence:
#         if i == last_verb:
#             full_verb.append(i)
#             add_aux = True
#         elif i.upos == 'AUX' and add_aux:
#             full_verb.append(i)
#         else:
#             add_aux = False
            
#     full_verb_form = ''.join(["する" if i.form == "為る" else i.form for i in full_verb])
    
#     if last_verb.lemma == "為る":
#         return "する", full_verb_form, full_verb
#     else:
#         return last_verb.lemma, full_verb_form, full_verb

# Jisho API version (incomplete, and will get rate limited)
def get_verb_type_jisho(verb):
    endpoint = f'https://jisho.org/api/v1/search/words?keyword={verb}'
    response = requests.get(endpoint)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    data = response.json()
    if data['data']:
        first_entry = data['data'][0]
        if 'senses' in first_entry and first_entry['senses']:
            verb_type = first_entry['senses'][0]['parts_of_speech'][0]
            first_word = verb_type.split()[0].upper()
            return first_word
    return None

def get_kana_form(verb):
    result = jam.lookup(verb)
    return result.entries[0].kana_forms[0]
    
    
# jamdict version 
def get_verb_type_jamdict(verb):
    result = jam.lookup(verb)
    print(result,verb)
    verb_type_string = result.entries[0].senses[0].to_dict()['pos'][0]
    verb_type = verb_type_string.split()[0].upper()

    
    if verb_type == "GODAN":
        consonant = verb_type_string.split()[3]
        consonant = consonant.strip("'")[0]
        verb_type = f"v5{consonant}"
    if verb_type == "ICHIDAN":
        verb_type = "v1"
    if verb_type == "KURU":
        verb_type = "vk"
    if verb_type == "SURU":
        verb_type = "vs-i"
        
    return verb_type

def get_inflection(verb, verb_tense):
    return inflections.inflect(verb, get_verb_type_jamdict(verb))[verb_tense]

def is_equal(verb1,verb2):
    converted_verb1 = converter.convert(verb1,n_best=10)
    converted_verb2 = converter.convert(verb2,n_best=10)
    union_set = set(converted_verb1).union(set(converted_verb2))
    return len(union_set) >= 1



#print(find_verb("食べさせる"))
#print(inflections.inflect("食べる", get_verb_type_jamdict("食べる")))

s=nlp("其國を治めんと欲する者は先づ其家を齊ふ")
print(s)
t=s[4]
print(t.id,t.form,t.lemma,t.upos,t.xpos,t.feats,t.head.id,t.deprel,t.deps,t.misc)

print(inflections.inflect("洗わせられる", "v1")["past"])
