# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:17:37 2018

@author: Rafael
"""

import csv
import matplotlib.pyplot as plt
from collections import Counter

#.csv file reading
def read_csv(file):
    with open(file, newline='',encoding="iso-8859-1") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        mat = [" ".join(row) for row in spamreader ]
    return mat

#convertion for matrix format
def convertion(note):
    text = ""
    for row in range(len(note)):
        if note[row] != " ":
            text = text+note[row]+" "
    text = text.split("@@;")[1:]
    for row in range(len(text)):
        text[row] = text[row].split(';@;')
        for i in range(len(text[row])):
            while "  " in text[row][i]:
                text[row][i] = text[row][i].replace("  "," ")
                text[row][i] = text[row][i].replace(" .",".")
        text[row][3] = (text[row][3] + ' ' + text[row][4]).replace('" "',' ')
        text[row].pop(4)
    return text

#vocabulary formation
def vocabulary(note):
    voc = {}
    for i in range(len(note)):
        for j in range(len(note[i])):
            text = note[i][j].lower()
            for char in text:
                if not char.isalpha() and char != " ":
                    text = text.replace(char,"")
            text = text.split()
            if voc == {}:
                voc[text[0]] = 1
            else:
                for word in range(len(text)):
                    if text[word] not in voc:
                        voc[text[word]] = voc[list(voc.keys())[-1]] + 1
    return voc

#free-text word count
def count_words(text):
    text = text.split(" ")
    return len(text)

#free-text phrase count
def phrases(text):
    p_lst = []
    if text == "" or text == " ":
        return p_lst
    elif "." not in text:
        return [text]
    else:
        for i in range(len(text)-1):
            if ((text[i-1].isalpha() or text[i-1] == ")") and text[i] == "." and text[i+1] == ' '):
                text = text[:i] + 'r2p7m45' + text[i+1:]
        text = text.lower()
        for char in text:
            if char not in "0123456789" and not char.isalpha() and char != " ":
                text = text.replace(char,"")
        p_lst = text.split("r2p7m45 ")
    return p_lst

#icd-9 to icd-10 mapping dictionary
def icd_dict(m):
    icd = {}
    for i in range(len(m)):
        a = m[i].split(";")
        if a[1] not in icd:
            icd[a[1]] = [a[2]]
        else:
            icd[a[1]].append(a[2])
    return icd

#icd-9 code extraction from col "Diagnósticos", code frequency and code number per discharge note
def codes(st,c_lst,cpn):
    s = st[:]
    s = s.split()
    icd = []
    for word in s:
        w = ""
        for i in range(len(word)):
            if word[i] in "0123456789":
                w = w + word[i]
        icd.append(w)
        if w not in c_lst and w != "":
            c_lst[w] = 1
        elif w in c_lst:
            c_lst[w] += 1
        for c in icd:
            if c == "":
                icd.remove("")
    cpn.append(len(icd))
    return icd

#icd-9 to icd-10 mapping
def icd9_to_10(lst,icd_d):
    l = lst[:]
    for i in range(len(l)):
        if l[i] not in icd_d:
            l[i] = "UNK"
        else:
            l[i] = icd_d[l[i]][0]
    return l

#icd-10 block extraction
def icd10_block(lst):
    l = lst[:]
    for i in range(len(l)):
        l[i] = l[i][0:3]
    return l

#icd-10 chapter extraction
def icd10_chap(lst):
    c_labels_cid = lst[:]
    for i in range(len(c_labels_cid)):
        if c_labels_cid[i] >= 'A00' and c_labels_cid[i] <= 'B99': 
            c_labels_cid[i] = 1 
        elif c_labels_cid[i] >= 'C00' and c_labels_cid[i] <= 'D48': 
            c_labels_cid[i] = 2
        elif c_labels_cid[i] >= 'D50' and c_labels_cid[i] <= 'D89': 
            c_labels_cid[i] = 3
        elif c_labels_cid[i] >= 'E00' and c_labels_cid[i] <= 'E90': 
            c_labels_cid[i] = 4
        elif c_labels_cid[i] >= 'F00' and c_labels_cid[i] <= 'F99': 
            c_labels_cid[i] = 5
        elif c_labels_cid[i] >= 'G00' and c_labels_cid[i] <= 'G99': 
            c_labels_cid[i] = 6
        elif c_labels_cid[i] >= 'H00' and c_labels_cid[i] <= 'H59': 
            c_labels_cid[i] = 7
        elif c_labels_cid[i] >= 'H60' and c_labels_cid[i] <= 'H95': 
            c_labels_cid[i] = 8
        elif c_labels_cid[i] >= 'I00' and c_labels_cid[i] <= 'I99': 
            c_labels_cid[i] = 9
        elif c_labels_cid[i] >= 'J00' and c_labels_cid[i] <= 'J99': 
            c_labels_cid[i] = 10
        elif c_labels_cid[i] >= 'K00' and c_labels_cid[i] <= 'K93': 
            c_labels_cid[i] = 11
        elif c_labels_cid[i] >= 'L00' and c_labels_cid[i] <= 'L99': 
            c_labels_cid[i] = 12
        elif c_labels_cid[i] >= 'M00' and c_labels_cid[i] <= 'M99': 
            c_labels_cid[i] = 13
        elif c_labels_cid[i] >= 'N00' and c_labels_cid[i] <= 'N99': 
            c_labels_cid[i] = 14
        elif c_labels_cid[i] >= 'O00' and c_labels_cid[i] <= 'O99': 
            c_labels_cid[i] = 15
        elif c_labels_cid[i] >= 'P00' and c_labels_cid[i] <= 'P96': 
            c_labels_cid[i] = 16
        elif c_labels_cid[i] >= 'Q00' and c_labels_cid[i] <= 'Q99': 
            c_labels_cid[i] = 17
        elif c_labels_cid[i] >= 'R00' and c_labels_cid[i] <= 'R99': 
            c_labels_cid[i] = 18
        elif c_labels_cid[i] >= 'S00' and c_labels_cid[i] <= 'T98': 
            c_labels_cid[i] = 19
        elif c_labels_cid[i] >= 'V01' and c_labels_cid[i] <= 'Y98': 
            c_labels_cid[i] = 20
        elif c_labels_cid[i] >= 'Z00' and c_labels_cid[i] <= 'Z99': 
            c_labels_cid[i] = 21
        elif c_labels_cid[i] == "UNK":
            c_labels_cid[i] = "UNK"
        else:
            c_labels_cid[i] = 22
    return c_labels_cid


"""

FUNCTION TESTING BELOW

"""

#ICD-9 to ICD-10 convertion table and HBA discharge notes
icd_map = read_csv("icd_forward.csv")
dn = read_csv("notas_ex_hba.csv")

#ICD-9 to ICD-10 map
icd_map = icd_dict(icd_map)
del icd_map["ICD9"]

#dataset in matrix format
dn = convertion(dn)
dn_fields = dn.pop(0)
dn_fields.pop(7)

#vocabulary formation
vocabulary = vocabulary(dn)

#statistics vector: vector containing sum of total number of words for each field and code list at stat_vec[8]
stat_vec_words = [0] * len(dn[0])
stat_vec_words[7] = {}
stat_vec_phrases = [0] * len(dn[0])
codes_per_note = []

#statistics loop
for i in range(len(dn)):
    for j in range(len(dn[i])):
        if j == 0 or j == 3 or j ==6 or j == 8 or j == 9 or j == 10 or j == 11:
            stat_vec_words[j] += count_words(dn[i][j])
            stat_vec_phrases[j] += len(phrases(dn[i][j]))
        elif j == 7:
            codes(dn[i][j],stat_vec_words[j],codes_per_note)
code_map = stat_vec_words.pop(7)
stat_vec_phrases.pop(7)
stat_vec_words = [x / len(dn) for x in stat_vec_words]
stat_vec_phrases = [x / len(dn) for x in stat_vec_phrases]

#icd mapping and chapter extraction
chap_freq = Counter(icd10_chap(icd10_block(icd9_to_10(list(code_map.keys()),icd_map))))

#code frequency bar chart
plt.bar(range(len(code_map)), [code_map[x] for x in sorted(code_map)], align='center')
plt.xticks(range(len(code_map)), sorted(code_map))

plt.show()

#chapter frequency: UNK are outside map codes
plt.bar(range(len(chap_freq)), chap_freq.values(), align='center')
plt.xticks(range(len(chap_freq)), list(chap_freq.keys()))

plt.show()

#word frequency by field
plt.bar(range(len(stat_vec_words)), stat_vec_words, align='center')
plt.xticks(range(len(dn_fields)), dn_fields, rotation = "vertical")

plt.show()

#phrase frequency by field
plt.bar(range(len(stat_vec_phrases)), stat_vec_phrases, align='center')
plt.xticks(range(len(dn_fields)), dn_fields, rotation = "vertical")

plt.show()

#converting dataset to .txt file
fields = [line[3] for line in dn]
fields = [phrases(line) for line in fields]
code_list = [icd9_to_10(codes(line[7],{},[]),icd_map) for line in dn]
    
i = len(fields)-1
while i >= 0:
    if len(fields[i]) >= 9:
        fields[i] = fields[i][0:9]
    else:
        if len(fields[i]) == 0:
            fields[i].append("")
        while len(fields[i]) < 9:
            fields[i].append("")
    if code_list[i][0] == 'UNK':
        code_list.pop(i)
        fields.pop(i)
    else:
        fields[i].insert(0,code_list[i][0])
        fields[i].append(code_list[i])
    i = i - 1

f = open('example_dataset.txt','w')

for _list in fields:
    for _string in range(len(_list)):
        if _string == 0 and _list[1] != "":
            f.write(str(_list[_string]) + ' <> 873heGKe7I ')
        elif _string == 10:
            f.write(str(_list[_string]) + '\n')
        elif _list[_string+1] == "" or _string+1 == 10:
            f.write(str(_list[_string]) + ' <> ')
        else:
            f.write(str(_list[_string]) + ' 873heGKe7I <> 873heGKe7I ') 

f.close()

print("\nScript over: example_dataset.txt overwritten!")