import math
import operator
from tqdm import tqdm

def read_file(filePath):
    allFile = []
    file = open(filePath,'r', encoding='utf-8')
    for line in file:
        allFile.append(line.lower())
    file.close()
    return allFile


file = read_file('metu.txt')
word_tag = [] #kelime + tag
for sentence in file:
    word_tag.append(sentence.split())

train_count = int(len(word_tag)*0.7)
train_word_tag = []
test_word_tag =[]
for i in range(len(word_tag)):
    if i <= train_count:
        train_word_tag.append(word_tag[i])
    else:
        test_word_tag.append(word_tag[i])


tag_dict = {}
for i in range(len(train_word_tag)):
    for j in range(len(train_word_tag[i])):
        word = train_word_tag[i][j].split('/')[0]
        tag = train_word_tag[i][j].split('/')[1]
        if tag not in tag_dict:
            tag_dict[tag] = {word: 1}
        else:
            if word not in tag_dict[tag]:
                tag_dict[tag].update({word: 1})
            else:
                tag_dict[tag][word] = tag_dict[tag].get(word) + 1


tag_count = {}
for key in tag_dict:
    tag_count[key] = sum(tag_dict[key].values())

emission_prob = {}
for i in range(len(train_word_tag)):
    for j in range(len(train_word_tag[i])):
        word = train_word_tag[i][j].split('/')[0]
        tag = train_word_tag[i][j].split('/')[1]
        emission = word + "/" + tag
        if tag in tag_dict:
            if word in tag_dict[tag]:
                emission_prob[emission] = tag_dict[tag].get(word) / tag_count[tag]
            else:
                emission_prob[emission] = 1 / tag_count[tag]


initial_count = {}
for i in range(len(train_word_tag)):
    start_tag = train_word_tag[i][0].split('/')[1]
    if start_tag in initial_count:
        initial_count[start_tag] = initial_count.get(start_tag) + 1
    else:
        initial_count[start_tag] = 1

initial_prob = {}
tags = []
for key in initial_count:
    initial_prob[key] = initial_count.get(key) / train_count
    tags.append(key)

transition_count = {}
for i in range(len(train_word_tag)):
    for j in range(len(train_word_tag[i])-1):
        first_tag = train_word_tag[i][j].split('/')[1]
        second_tag = train_word_tag[i][j+1].split('/')[1]
        bigram_tag = first_tag + "/" + second_tag
        if bigram_tag in transition_count:
            transition_count[bigram_tag] = transition_count.get(bigram_tag) + 1
        else:
            transition_count[bigram_tag] = 1


transition_prob = {}
for key in transition_count:
    sec_tag = key.split('/')[1]
    transition_prob[key] = transition_count.get(key) / tag_count[sec_tag]


print()
print("Processing...")
print()

#############
###VITERBI###
#############

# Output File configuration
filename = "output.txt"
output_file = open(filename, 'w', encoding='utf-8')


correct_count = 0
number_of_words = 0

viterbi_dict = {}
viterbi_list = []
viterbi_total_list = []
tag_list_estimation = []
total_tag_list_estimation = []

for i in tqdm(test_word_tag):
    for j in tags:
        word = i[0].split('/')[0]
        pair = word + "/" + j
        if pair not in emission_prob:
            tag_p = initial_prob.get(j) * (1 / tag_count.get(j))
        else:
            tag_p = initial_prob.get(j) * emission_prob.get(word + "/" + j)
        temp_dict = {j: tag_p}
        viterbi_dict.update(temp_dict)
        temp_dict = {}

    viterbi_list.append(viterbi_dict)
    viterbi_dict = {}
    for k in range(1, len(i)):
        for j in tags:
            word = i[k].split('/')[0]
            pair = word + "/" + j
            if pair not in emission_prob:
                if k == 1:
                    tag_p = {key: value * (1 / tag_count.get(j)) for key, value in viterbi_list[k - 1].items()}
                else:
                    tag_p = {key : value[1] * (1 / tag_count.get(j)) for key, value in viterbi_list[k-1].items()}
            else:
                if k == 1:
                    tag_p = {key: value * emission_prob.get(word + "/" + j) for key, value in
                             viterbi_list[k - 1].items()}
                else:
                    tag_p = {key : value[1] * emission_prob.get(word + "/" + j) for key, value in viterbi_list[k-1].items()}
            sorted_tag_p = sorted(tag_p.items(), key=operator.itemgetter(1))
            max = 0.0
            for l in tags:
                if l + "/" + j in transition_prob.keys():
                    temp_max = transition_prob.get(l + "/" + j) * sorted_tag_p[-1][1]
                    if temp_max > max:
                        max = temp_max
                else:
                    temp_max = (1 / tag_count.get(j)) * sorted_tag_p[-1][1]
                    if temp_max > max:
                        max = temp_max

            temp_dict = {j: [sorted_tag_p[-1][0], max]}
            viterbi_dict.update(temp_dict)
            temp_dict = {}
            tag_p = {}

        viterbi_list.append(viterbi_dict)
        viterbi_dict = {}


    # Backpropagation
    viterbi_list.reverse()
    sorted_temp = sorted(viterbi_list[0].items(), key=operator.itemgetter(1))
    tag_list_estimation.append(sorted_temp[-1][0])
    previous_key = sorted_temp[-1][1][0]
    tag_list_estimation.append(previous_key)
    for x in range(1, len(viterbi_list)-1):
        previous_key = viterbi_list[x].get(previous_key)[0]
        tag_list_estimation.append(previous_key)
    tag_list_estimation.reverse()                                           # estimated tags of the sentence
    tag_list_test = [y.split('/')[1] for y in i]                            # real tags of the sentence
    for x in range(len(tag_list_test)):
        if tag_list_estimation[x] == tag_list_test[x]:
            correct_count += 1

    number_of_words += len(i)

    # Writing Output File
    sentence_estimation = ""
    word_list_test = [y.split('/')[0] for y in i]
    for x in range(len(word_list_test)):
        sentence_estimation += word_list_test[x] + '/' + tag_list_estimation[x] + ' '
    sentence_estimation += '\n'
    output_file.write(sentence_estimation)

    viterbi_list = []


print()
print("Accuracy: ", (correct_count / number_of_words))
print("Output.txt file has written!")
print("Done!")
