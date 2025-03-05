import tense_check as tc

import pprint

pprint.pprint(tc.find_verb("母は弟に野菜を食べさせられる。"))

pprint.pprint(tc.find_verb("歌わせる。"))

pprint.pprint(tc.find_verb("歌わされる。"))

pprint.pprint(tc.find_verb("歌わせられる。"))


import spacy

nlp = spacy.load("ja_core_news_sm")

doc = nlp("歌わせられる、歌わされる、妹は母に皿を洗わせられる")

for token in doc:
    print(token.text,token.lemma_,token.pos_,token.morph,token.is_stop,token.tag_)
    print(type(token))
    print(type(token.text))
    print(token.morph.to_dict())



