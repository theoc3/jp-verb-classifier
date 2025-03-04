import tense_check as tc

causative_baseline_test = ["母は弟に野菜を食べさせた。", "先生は生徒に宿題をやらせた。", "父は私に車を運転させた。", "兄は妹に犬の世話をさせた。", "先生は学生に本を読ませた。", "友達は私に歌を歌わせた。", "母は子供に部屋を掃除させた。", "先輩は後輩に荷物を運ばせた。", "店長は店員にドアを開けさせた。", "先生は生徒に作文を書かせた。"]

#print(tc.find_verb("見つけた人は運転させた")[-1])

for sentence in causative_baseline_test:
    print(sentence)
    #print(tc.find_verb(sentence)[-1][0],tc.find_verb(sentence)[-1][1])
    print(tc.get_inflection(tc.find_verb(sentence)[-1][0],"past-causative"))
    
    
    
    #print(tc.find_verb(sentence)[-1])

l1 = ["話される","話せる","話させられる"]
l2 = ["待たれる","待たせる","待たさせられる"]
l3 = ["食べられる","食べさせる","食べさせられる"]  

 
# test1 = tc.find_verb("私は話される。")
# test2 = tc.find_verb("マジで変なバカにした先生が学生たちに宿題をさせた。")
# print(test1)
# print(test2)

# print(tc.get_verb_type(test1[0]))
# print(tc.get_verb_type(test2[0]))

# print(tc.get_inflection(test2[0], "past-passive"))

# print(tc.get_inflection("食べる","potential"))
# print(tc.get_inflection("食べる","passive"))