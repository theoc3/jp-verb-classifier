# lemmization, POS tagging, and dependency parsing
import unidic2ud
nlp=unidic2ud.load("kindai")

# dictionary lookup
import jamdict
jam = jamdict.Jamdict()

import requests
import json

# conjugation script
import inflections
    
def find_verb(sentence):
    last_verb = None
    nlp_sentence = nlp(sentence)
    for i in nlp_sentence:
        if i.upos == 'VERB':
            last_verb = i
    add_aux = False
    full_verb = []
    
    for i in nlp_sentence:
        if i == last_verb:
            full_verb.append(i)
            add_aux = True
        elif i.upos == 'AUX' and add_aux:
            full_verb.append(i)
        else:
            add_aux = False
            
    full_verb_form = ''.join(["する" if i.form == "為る" else i.form for i in full_verb])
    
    if last_verb.lemma == "為る":
        return "する", full_verb_form, full_verb
    else:
        return last_verb.lemma, full_verb_form, full_verb

# Jisho API version (may get rate limited)
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

def get_verb_type(verb):
    result = jam.lookup(verb)
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
    return inflections.inflect(verb, get_verb_type(verb))[verb_tense]

l1 = ["話される","話せる","話させられる"]
l2 = ["待たれる","待たせる","待たさせられる"]
l3 = ["食べられる","食べさせる","食べさせられる"]  

 
test1 = find_verb("私は話される。")
test2 = find_verb("マジで変なバカにした先生が学生たちに宿題をさせた。")
print(test1)
print(test2)

print(get_verb_type(test1[0]))
print(get_verb_type(test2[0]))

print(inflections.inflect(test1[0], get_verb_type(test1[0])))

print(get_inflection(test2[0], "past-passive"))

print(get_inflection("食べる","potential"))
print(get_inflection("食べる","passive"))



#t=s[8]
#print(t.id,t.form,t.lemma,t.upos,t.xpos,t.feats,t.head.id,t.deprel,t.deps,t.misc)