import tense_check as tc

causative_baseline_test = ["母は弟に野菜を食べさせた。", "先生は生徒に宿題をやらせた。", "父は私に車を運転させた。", "兄は妹に犬の世話をさせた。", "先生は学生に本を読ませた。", "友達は私に歌を歌わせた。", "母は子供に部屋を掃除させた。", "先輩は後輩に荷物を運ばせた。", "店長は店員にドアを開けさせた。", "先生は生徒に作文を書かせた。"]

passive_baseline_test = ['私は先生に怒られる。', '彼は上司に文句を言われる。', '子供は母に叱られる。', '彼女は友達に笑われる。', '生徒たちは先生に名前を呼ばれる。', '私は兄にゲームを壊される。', '彼はチームメイトにパスを取られる。', '私たちは監督に作戦を知られる。', '彼は親に日記を読まれる。', '妹は母に手紙を見られる。']

causative_passive_baseline_test = ["私は先生に宿題をやらされる。", "彼は上司に残業をさせられる。", "子供は親に野菜を食べさせられる。", "彼女は友達に無理やり歌を歌わせられる。", "生徒たちは先生に長い作文を書かせられる。", "私は兄に部屋を掃除させられる。", "彼はチームメイトに試合に出させられる。", "私たちは監督に毎日走らせられる。", "彼は親にピアノを練習させられる。", "妹は母に皿を洗わせられる。"]



#print(tc.find_verb("見つけた人は運転させた")[-1])

for sentence in causative_baseline_test:
    print(sentence)
    #print(tc.find_verb(sentence)[-1][0],tc.find_verb(sentence)[-1][1])
    verb = tc.find_verb(sentence)[-1]
    conjugated_verb = tc.get_inflection(verb[0],"past-causative")
    print("verb extracted:",verb[1])
    print("manual conjugation check:",conjugated_verb)
    print(tc.is_equal(verb[1],conjugated_verb))
   
for sentence in passive_baseline_test:
    print(sentence)
    #print(tc.find_verb(sentence)[-1][0],tc.find_verb(sentence)[-1][1])
    verb = tc.find_verb(sentence)[-1]
    conjugated_verb = tc.get_inflection(verb[0],"passive")
    print("verb extracted:",verb[1])
    print("manual conjugation check:",conjugated_verb)
    print(tc.is_equal(verb[1],conjugated_verb))
    
for sentence in causative_passive_baseline_test:
    print(sentence)
    #print(tc.find_verb(sentence)[-1][0],tc.find_verb(sentence)[-1][1])
    verb = tc.find_verb(sentence)[-1]
    conjugated_verb = tc.get_inflection(verb[0],"causative-passive")
    print("verb extracted:",verb[1])
    print("manual conjugation check:",conjugated_verb)
    print(tc.is_equal(verb[1],conjugated_verb))
    
    
    
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