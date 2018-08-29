# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:12:33 2018

@author: ext01
"""
#%%
from __future__ import print_function
import numpy as np
np.random.seed(1337)

import codecs
import re
import csv
import operator
import matplotlib
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from collections import Counter
from collections import OrderedDict
from heatmap import heatmap, annotate_heatmap

print('Loading data...')

texts = [line.split('\t') for line in [line for line in codecs.open('dataset_hba.txt', encoding="iso-8859-1")]]
fields = texts.pop(0)
texts = [line for line in texts if len(line) == 121 and not (line[15] == '' or line[15] == ' ')]

def clean(l_cid):
    n_o_s = list(l_cid)
    while True:
        l_flatten = [line[0] for line in n_o_s if line != []]
        l_freq = dict((k,v) for (k,v) in Counter(l_flatten).items())
        o_s = set([k for (k,v) in l_freq.items() if v == 1])
        print(len(o_s))
        if len(o_s) == 0: break
        for i in range(len(n_o_s)):
            if n_o_s[i] != [] and n_o_s[i][0] in o_s:
                if len(n_o_s[i]) > 1: n_o_s[i] = n_o_s[i][1:]
                else: n_o_s[i] = ['']
    return n_o_s

def codes(st):
    s = st[:].replace('%','')
    s = s.split()
    icd = []
    for word in s:
        w = ""
        for i in range(len(word)):
            if word[i] in "0123456789" and word[-1] == '-' or ((word[i] == 'V' or word[i] == 'E') and len(word) >= 3 and word[1] in "0123456789"):
                w = w + word[i]
        icd.append(w)
        for c in icd:
            if c == "":
                icd.remove("")
    return icd

def icd9_chap(lst):
    c_labels_cid = lst[:]
    for i in range(len(c_labels_cid)):
        if c_labels_cid[i] >= '001' and c_labels_cid[i] <= '139': 
            c_labels_cid[i] = 1 
        elif c_labels_cid[i] >= '140' and c_labels_cid[i] <= '239': 
            c_labels_cid[i] = 2
        elif c_labels_cid[i] >= '240' and c_labels_cid[i] <= '279': 
            c_labels_cid[i] = 3
        elif c_labels_cid[i] >= '280' and c_labels_cid[i] <= '289': 
            c_labels_cid[i] = 4
        elif c_labels_cid[i] >= '290' and c_labels_cid[i] <= '319': 
            c_labels_cid[i] = 5
        elif c_labels_cid[i] >= '320' and c_labels_cid[i] <= '389': 
            c_labels_cid[i] = 6
        elif c_labels_cid[i] >= '390' and c_labels_cid[i] <= '459': 
            c_labels_cid[i] = 7
        elif c_labels_cid[i] >= '460' and c_labels_cid[i] <= '519': 
            c_labels_cid[i] = 8
        elif c_labels_cid[i] >= '520' and c_labels_cid[i] <= '579': 
            c_labels_cid[i] = 9
        elif c_labels_cid[i] >= '580' and c_labels_cid[i] <= '629': 
            c_labels_cid[i] = 10
        elif c_labels_cid[i] >= '630' and c_labels_cid[i] <= '679': 
            c_labels_cid[i] = 11
        elif c_labels_cid[i] >= '680' and c_labels_cid[i] <= '709': 
            c_labels_cid[i] = 12
        elif c_labels_cid[i] >= '710' and c_labels_cid[i] <= '739': 
            c_labels_cid[i] = 13
        elif c_labels_cid[i] >= '740' and c_labels_cid[i] <= '759': 
            c_labels_cid[i] = 14
        elif c_labels_cid[i] >= '760' and c_labels_cid[i] <= '779': 
            c_labels_cid[i] = 15
        elif c_labels_cid[i] >= '780' and c_labels_cid[i] <= '799': 
            c_labels_cid[i] = 16
        elif c_labels_cid[i] >= '800' and c_labels_cid[i] <= '999': 
            c_labels_cid[i] = 17
        elif c_labels_cid[i] >= 'V01' and c_labels_cid[i] <= 'V91': 
            c_labels_cid[i] = 18
        else: 
            c_labels_cid[i] = 19
    return c_labels_cid

def clean_num(sent):
    aux = sent.split(' ')
    ints = []
    for word in aux:
        numDigits = 0
        for char in word:
            if char.isdigit(): numDigits += 1
        if len(word) > 0: digitFraction = numDigits / float(len(word))
        else: digitFraction = 0
        if word.isdigit(): ints.append(word)
        elif digitFraction > 0.0: ints.append(word)
    for num in ints: sent = sent.replace(num,'<NUMBER>', 1)
    return sent

def count_phrases(text):
    p_lst = []
    if text == "" or text == " ":
        return p_lst
    else:
        for i in range(len(text)-1):
            if ((text[i-1].isalpha() or text[i-1] == ")") and (text[i] == "." or text[i] == ";") and text[i+1] == ' '):
                text = text[:i] + 'r2fa7m45ir' + text[i+1:]
        p_lst = text.split("r2fa7m45ir ")
    return p_lst

print('Mapping ICD-9 to ICD-10 codes...')

labels_cid = clean([codes(line[14]) for line in texts])
labels_flatten = [item for sublist in labels_cid for item in sublist]
blocks = sorted(list(set([item[:3] for item in labels_flatten])))
chap = icd9_chap([x[:3] for x in labels_flatten])

print('\n- Statistical characterization of the dataset:')
print('  - Distinct ICD-9 codes: %s' % len(set(labels_flatten)))
print('  - Distinct ICD-9 blocks: %s' % len(blocks))
print('  - Distinct ICD-9 chapters: %s' % len(set(chap)))

#%%
print('- Department provenance:')
dep = [line[71] for line in texts]

for i in range(len(dep)):
    if dep[i] == 'Medicina':
        dep[i] = (texts[i][71]+' '+texts[i][72]).replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
        if dep[i] == 'Medicina ': dep[i] = 'Medicina Não especificado'
    elif dep[i] == 'Cirurgia':
        dep[i] = (texts[i][71]+' '+texts[i][73]).replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
        if dep[i] == 'Cirurgia ': dep[i] = 'Cirurgia Não especificado'
    elif dep[i] == ' ' or dep[i] == '':
        dep[i] = 'Não especificado'

dep_options = sorted(list(set(dep)))
chap_set = list(set([x for x in chap if x != 19]))
prov_chap = np.zeros((len(dep_options),len(chap_set)))

for aux in range(len(dep)):
    i = dep_options.index(dep[aux])
    for j in icd9_chap([x[:3] for x in labels_cid[aux]]):
        if j < 19: prov_chap[i][j-1] += 1

fig, ax = plt.subplots(figsize=(20, 20))

im, cbar = heatmap(prov_chap, dep_options, chap_set, ax=ax,
                   cmap="Reds", cbarlabel="Prevalence")
texts = annotate_heatmap(im, valfmt="{x:.0f}")

plt.ylabel('Department', fontsize='large')
ax.set_xlabel('ICD-9 chapters', fontsize='large')    
ax.xaxis.set_label_position('top') 

fig.tight_layout()
fig.savefig('heatmap_dep.png',dpi=400)

#%%
print('- Age Groups:')
idade = [int(line[2]) for line in texts]

for i in range(len(idade)):
    if idade[i] < 5: idade[i] = 0
    elif idade[i] < 15: idade[i] = 1
    elif idade[i] < 25: idade[i] = 2
    elif idade[i] < 45: idade[i] = 3
    elif idade[i] < 65: idade[i] = 4
    else: idade[i] = 5
    
age_options = sorted(list(set(idade)))
ag = ['0 - 4', '5 - 14', '15 - 24', '25 - 44', '45 - 64', '> 65']
chap_set = list(set([x for x in chap if x != 19]))
age_groups = np.zeros((len(age_options),len(chap_set)))

for aux in range(len(idade)):
    i = age_options.index(idade[aux])
    for j in icd9_chap([x[:3] for x in labels_cid[aux]]):
        if j < 19: age_groups[i][j-1] += 1

fig, ax = plt.subplots(figsize=(15, 4))

im, cbar = heatmap(age_groups, ag, chap_set, ax=ax,
                   cmap="Reds", cbarlabel="Prevalence")
texts = annotate_heatmap(im, valfmt="{x:.0f}")

plt.ylabel('Age groups', fontsize='large')
ax.set_xlabel('ICD-9 chapters', fontsize='large')    
ax.xaxis.set_label_position('top') 

fig.tight_layout()
fig.savefig('heatmap_age.png',dpi=400)

#%%
print('\n- ICD-10 full-code statistics:')

labels_freq = Counter(labels_flatten)
labels_freq = sorted(labels_freq.items(), key=operator.itemgetter(1))[::-1]
labels_freq = [tup for tup in labels_freq if tup[1] > 600 and tup[0] != '']
labels_freq = OrderedDict(labels_freq)

labels_fst = [line[0] for line in labels_cid if line != []]
l_fst = OrderedDict((k,0) for (k,v) in labels_freq.items())
for code in labels_fst:
    if code in l_fst: l_fst[code] += 1

ind = range(len(labels_freq))
keys = list(labels_freq.keys())
fst = list(l_fst.values())
cid = [list(labels_freq.values())[i]-fst[i] for i in ind]
    
p1 = plt.bar(ind, cid, color='teal')
p2 = plt.bar(ind, fst, color='firebrick', bottom=cid)

plt.xticks(ind, keys, rotation = "vertical")
plt.title('Number of occurences of the 50 more common ICD-9 codes in the dataset')
plt.xlabel('ICD-9 codes', fontsize='large')
plt.ylabel('Number of occurences in the dataset', fontsize='large')
x1,x2,y1,y2 = plt.axis()
plt.axis((-2,51,0,5000))

plt.legend((p1[0],p2[0]), ('Not main code entries','Main code entries'), fontsize='xx-large')

fig = plt.gcf()
fig.set_size_inches(20.0, 10.5)
fig.savefig('labels_freq_50.png',dpi=400)

#%%
per_note = [len(lst) for lst in labels_cid if lst != [] and lst != ['']]
per_note_blk = [len(lst) for lst in [list(set([code[:3] for code in line])) for line in labels_cid if line != [] and line != ['']]]
per_note_chap = [len(lst) for lst in [list(set(icd9_chap(line))) for line in labels_cid if line != [] and line != ['']]]

print('\n  - ICD-9 codes per instance: %.3f' % np.mean(per_note))
print('  - ICD-9 blocks per instance: %.3f' % np.mean(per_note_blk))
print('  - ICD-9 chapters per instance: %.3f' % np.mean(per_note_chap))
print('\n  - Number of instances: %s' % len([x for x in labels_cid if x != [] and x != ['']]))
print('  - Number of instances with multiple ICD-9 codes: %s' % len([l for l in per_note if l > 1]))

per_note = OrderedDict(sorted(Counter(per_note).items()))
pnblk = dict((k,0) for (k,v) in per_note.items())
for l in per_note_blk: pnblk[l] += 1
pnchap = dict((k,0) for (k,v) in per_note.items())
for l in per_note_chap: pnchap[l] += 1

fig = plt.figure()

x = list(per_note.keys())

y = list(per_note.values())
z = list(pnblk.values())
k = list(pnchap.values())

x1 = [i-0.25 for i in x]
x3 = [i+0.25 for i in x]

ax = plt.subplot(111)
p1 = ax.bar(x1, y, width=0.2, color='maroon', align='center')
p2 = ax.bar(x, z, width=0.2, color='firebrick', align='center')
p3 = ax.bar(x3, k, width=0.2, color='salmon', align='center')
ax.set_ylabel('Number of instances', fontsize='x-large')
ax.set_xlabel('Number of occurences', fontsize='x-large')
ax.set_title('Number of ICD-9 full-codes, blocks and chapters per note in the dataset')
ax.set_xticks(list(range(1,max(per_note.keys())+1,1)))
x1,x2,y1,y2 = ax.axis()
ax.axis((-0.5,28.5,0,100000))
ax.legend((p1[0], p2[0], p3[0]), ('ICD-9 Codes','ICD-9 Blocks','ICD-9 Chapters'), fontsize='xx-large')

fig.set_size_inches(20.0, 10.5)
fig.savefig('code_per_note.png',dpi=400)

#%%
print('\n- ICD-9 chapter statistics:')

chap_freq = Counter((chap))
chap_freq = dict((k,v) for (k,v) in chap_freq.items() if k != 19)
chap_freq = OrderedDict(sorted(chap_freq.items()))

plt.bar(range(len(chap_freq)), chap_freq.values(), color='firebrick', align='center')
plt.xticks(range(len(chap_freq)), list(chap_freq.keys()))
plt.title('ICD-9 chapter frequency in the dataset')
plt.xlabel('ICD-9 chapters')
plt.ylabel('Number of occurences in the dataset')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1,18,0,40000))

fig = plt.gcf()
fig.set_size_inches(20.0, 10.5)
fig.savefig('chap_freq.png',dpi=400)

#%%
print('- Textual statistics:')
fields[120] = 'REQUISICAO_MEIO_TRANSPORTE'

mot_int = [line[4].replace('<>','') for line in texts]
res_clinic = [sent_tokenize(clean_num((line[8] + ' ' + line[9]).replace('%%%%%',' ').replace('<>',''))) for line in texts]
diag_alta = [[clean_num(re.sub(r'[^\w\s]','',diag)) for diag in line if diag != ''] for line in [field[15].replace('%%%%%','\n').replace('<>','').split('\n') for field in texts]]

input_text = [item for sublist in [diag_alta, mot_int, res_clinic] for item in sublist]
#input_text = res_clinic

stat_vec_words = [sum([len(sent.split(' ')) for sent in line]) for line in input_text]
stat_vec_phrases = [len(x) for x in input_text]
w_per_s = [0] * len(texts)

for i in range(len(texts)):
    if stat_vec_phrases[i] == 0: w_per_s[i] = 0
    else: w_per_s[i] = stat_vec_words[i]/stat_vec_phrases[i]

print('\n  - Sentences per instance(Average): %.3f' % np.mean(np.array(stat_vec_phrases)))
print('  - Sentences per instance (90 percentile): %.3f' % np.percentile(np.array(stat_vec_phrases), 90))
print('  - Words per instance (Average): %.3f' % np.mean(np.array(stat_vec_words)))
print('  - Words per sentence (Average): %.3f' % np.mean(np.array(w_per_s)))
print('  - Words per sentence (90 percentile): %.3f' % np.percentile(np.array(w_per_s), 90))

#%%
print('- Unfilled fields percentage:')

unfilled = [0] * len(fields)

for line in texts:
    for i in range(len(line)):
        if line[i] == '' or line[i] == ' ': unfilled[i] += 1
        
unfilled = [item*100/len(texts) for item in unfilled]

plt.bar(range(len(unfilled)), unfilled, color='firebrick', align='center')
plt.xticks(range(len(fields)), fields, rotation = "vertical")
plt.title('Percentage of unfilled instances by discharge summary field')
plt.xlabel('Discharge summary field')
plt.ylabel('Unfilled percentage')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1,121,0,100))

fig = plt.gcf()
fig.set_size_inches(30.0, 26.0)
fig.savefig('unfilled.png',dpi=300)

#%%
stat_vec_words = [0] * len(fields)
stat_vec_phrases = [0] * len(fields)
unfilled = [0] * len(fields)
for line in texts:
    for i in range(len(line)):
        if line[i] == '' or line[i] == ' ': unfilled[i] += 1
        stat_vec_phrases[i] += len(sent_tokenize(line[i]))
        stat_vec_words[i] += len(line[i].split(' '))

print('\n  - Sentences per instance: %.3f' % (sum(stat_vec_phrases[7:9])/(len(texts))))
print('  - Words per instance: %.3f' % (sum(stat_vec_words[7:9])/len(texts)))
print('  - Words per sentence: %.3f' % (sum(stat_vec_words[7:9])/sum(stat_vec_phrases[7:9])))

stat_vec_words = [int(round(x / len(texts))) for x in stat_vec_words]
stat_vec_phrases = [int(round((x / len(texts)))) for x in stat_vec_phrases]

plt.bar(range(len(stat_vec_words)), stat_vec_words, align='center')
plt.xticks(range(len(fields)), fields, rotation = "vertical")
plt.title('Average word count by discharge summary field')
plt.xlabel('Discharge summary field')
plt.ylabel('Word count')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1,121,0,120))

fig = plt.gcf()
fig.set_size_inches(30.0, 26.0)
fig.savefig('word_count.png',dpi=300)

#%%
plt.bar(range(len(stat_vec_phrases)), stat_vec_phrases, align='center')
plt.xticks(range(len(fields)), fields, rotation = "vertical")
plt.title('Average phrase count by discharge summary field')
plt.xlabel('Discharge summary field')
plt.ylabel('Phrase count')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1,121,0,7))

fig = plt.gcf()
fig.set_size_inches(30.0, 26.0)
fig.savefig('phrase_count.png',dpi=300)

