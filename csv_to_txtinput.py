# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:20:12 2018

@author: Rafael
"""
#%%
from __future__ import print_function
import numpy as np
np.random.seed(1337)

import re
import codecs
from collections import defaultdict
from nltk import sent_tokenize
from collections import Counter

#FUNCTIONS
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

def phrases(text):
    p_lst = []
    if text == "" or text == " ":
        return p_lst
    else:
        for i in range(len(text)-1):
            if ((text[i-1].isalpha() or text[i-1] == ")") and (text[i] == "." or text[i] == ";") and text[i+1] == ' '):
                text = text[:i] + 'r2fa7m45ir' + text[i+1:]
        p_lst = text.split("r2fa7m45ir ")
    return p_lst

#%%
#HBA DISCHARGE NOTES
print('Loading data...')

texts = [line.split('\t') for line in [line for line in codecs.open('dataset_hba.txt', encoding="iso-8859-1")]]
fields = texts.pop(0)

texts = [line for line in texts if len(line) == 121 and not (line[15] == '' or line[15] == ' ')]
#%%
print('Mapping ICD-9 codes...')
labels_cid_pre = [codes(line[14]) for line in texts]
labels_cid = clean(labels_cid_pre)

#%%
print('Preparing fields...')

idade = [line[2] for line in texts]
mot_int = [line[4].replace('<>','') for line in texts]
res_clinic = [sent_tokenize(clean_num((line[8] + ' ' + line[9]).replace('%%%%%',' ').replace('<>',''))) for line in texts]
diag_alta = [[clean_num(re.sub(r'[^\w\s]','',diag)) for diag in line if diag != ''] for line in [field[15].replace('%%%%%','\n').replace('<>','').split('\n') for field in texts]]
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

#%%
        
for i in range(len(res_clinic)):
    if len(res_clinic[i]) > 25: res_clinic[i] = res_clinic[i][:25]
    if len(diag_alta[i]) > 9: diag_alta[i] = diag_alta[i][:9]
    mot_int[i] = '873heGKe7I ' + mot_int[i] + ' 873heGKe7I'
    for j in range(len(res_clinic[i])):
        res_clinic[i][j] = '873heGKe7I ' + res_clinic[i][j] + ' 873heGKe7I'
        if res_clinic[i] == '873heGKe7I  873heGKe7I' or res_clinic[i] == '873heGKe7I   873heGKe7I': res_clinic[i] = ''
    for j in range(len(diag_alta[i])):
        diag_alta[i][j] = '873heGKe7I ' + diag_alta[i][j] + ' 873heGKe7I'
        if diag_alta[i] == '873heGKe7I  873heGKe7I' or diag_alta[i] == '873heGKe7I   873heGKe7I': diag_alta[i] = ''
    while len(res_clinic[i]) < 25:
        res_clinic[i].append('')
    while len(diag_alta[i]) < 9:
        diag_alta[i].append('')

#%%
print('Preparing discharge summaries file format...')

texts_aux = list(res_clinic)
for i in range(len(res_clinic)):
    if labels_cid[i] != [''] and len(labels_cid[i]) != 0:
        texts_aux[i] = labels_cid[i][0] + ' <> ' + diag_alta[i][0] + ' <> ' + diag_alta[i][1] + ' <> ' + diag_alta[i][2] + ' <> ' + diag_alta[i][3] + ' <> ' + diag_alta[i][4] + ' <> ' + diag_alta[i][5] + ' <> ' + diag_alta[i][6] + ' <> ' + diag_alta[i][7] + ' <> ' + diag_alta[i][8] + ' <> ' + mot_int[i] + ' <> ' + res_clinic[i][0] + ' <> ' + res_clinic[i][1] + ' <> ' + res_clinic[i][2] + ' <> ' + res_clinic[i][3] + ' <> ' + res_clinic[i][4] + ' <> ' + res_clinic[i][5] + ' <> ' + res_clinic[i][6] + ' <> ' + res_clinic[i][7] + ' <> ' + res_clinic[i][8] + ' <> ' + res_clinic[i][9] + ' <> ' + res_clinic[i][10] + ' <> ' + res_clinic[i][11] + ' <> ' + res_clinic[i][12] + ' <> ' + res_clinic[i][13] + ' <> ' + res_clinic[i][14] + ' <> ' +  res_clinic[i][15] + ' <> ' +  res_clinic[i][16] + ' <> ' + res_clinic[i][17] + ' <> ' + res_clinic[i][18] + ' <> ' + res_clinic[i][19] + ' <> ' + res_clinic[i][20] + ' <> ' + res_clinic[i][21] + ' <> ' + res_clinic[i][22] + ' <> ' + res_clinic[i][23] + ' <> ' + res_clinic[i][24] + ' <> ' + idade[i] + ' <> ' + dep[i] + ' <> ' + str(labels_cid[i])
        texts_aux[i] = texts_aux[i] + '\r'

#%%
print('Writing output...')

f = open('dataset_example_hba_full.txt','w')

for _list in texts_aux:
    if isinstance(_list,str): f.write(_list)
    
f.close()

#%%
#MIMIC-III DISCHARGE NOTES
print('Loading MIMIC-III data...')

texts = [line.split(',',10) for line in codecs.open('dataset_mimic.txt', encoding="iso-8859-1")]

texts = [[line[2],line[10]] for line in texts if ('History of Present Illness:'.lower() in line[10].lower() or 'History of the Present Illness:'.lower() in line[10].lower()) and ('Final Diagnoses:'.lower() in line[10].lower() or 'Discharge Diagnosis:'.lower() in line[10].lower() or 'Discharge Diagnoses:'.lower() in line[10].lower())]

fields = ['Admission Date:','Date:','Discharge Date:','Date of Birth:','Sex:','Service:','Allergies:','Attending:','Admitting Diagnosis:','Admission Diagnosis:','Admitting Diagnoses:','Admission Diagnoses:','Reason for Admission:','Chief Complaint:','Major Surgical or Invasive Procedure:','History of Present Illness:','History of the Present Illness:','Past Medical History:','Social History:','Family History:','Physical Exam:','Pertinent Results:','Pertinent Laboratory Data:','Laboratory studies on discharge:','Brief Hospital Course:','Concise Summary of Hospital Course:','Medications on Admission:','Discharge Medications:','Discharge Disposition:','Condition on Discharge:','Facility:','Discharge Diagnosis:','Final Diagnoses:','Discharge Diagnoses:','Discharge diagnoses:','Discharge Condition:','Discharge Instructions:','Discharge Status:','Patient instructions:','Followup Instructions:','Primary Care:','Completed by:']
for i in range(len(texts)):
    for field in fields:
        texts[i][1] = texts[i][1].replace(field,'1y3g4jhk2g'+field).replace(field.upper(),'1y3g4jhk2g'+field)
    texts[i][1] = texts[i][1].split('1y3g4jhk2g')
    
dep = [line[1][0] for line in texts] 
res_clinic = [line[1][0] for line in texts] 
mot_int = [line[1][0] for line in texts] 
diag_alta = [line[1][0] for line in texts] 

for i in range(len(texts)):
    for j in range(len(texts[i][1])): 
        if 'Service:' in texts[i][1][j]:
            dep[i] = re.sub(r'[^\w\s]','',texts[i][1][j].replace('Service: ','').replace('<>','').upper()).replace(' ','')
        if ('History of Present Illness:' in texts[i][1][j]) or ('History of the Present Illness:' in texts[i][1][j]):
            res_clinic[i] = sent_tokenize(clean_num(texts[i][1][j].replace('History of Present Illness: ','').replace('History of the Present Illness: ','').replace('<>','')))
        if 'Chief Complaint:' in texts[i][1][j]:
            mot_int[i] = texts[i][1][j].replace('Chief Complaint: ','').replace('<>','')
        if ('Discharge Diagnosis:' in texts[i][1][j]) or ('Final Diagnoses:' in texts[i][1][j]) or ('Discharge Diagnoses:' in texts[i][1][j]):
            diag_alta[i] = re.sub('\d','\n',texts[i][1][j].replace('"','').replace('Discharge Diagnosis: ','').replace('Discharge Diagnoses: ','').replace('Discharge diagnoses: ','').replace('Final Diagnoses: ','').replace('<>','')).split('\n')

#%%
print('Mapping ICD-9 codes...')

codes = [line.split(',') for line in codecs.open('dataset_mimic_codes.txt', encoding="utf-8")]
codes = [[line[2], line[4].replace('\r\n','').replace('"','')] for line in codes]
diagnoses_icd = defaultdict(list)
for hadm_id, code in codes:
    diagnoses_icd[hadm_id].append(code)

for i in range(len(texts)):
    texts[i][0] = diagnoses_icd[texts[i][0]] 

labels_cid_no = [line[0] for line in texts]
labels_cid = clean([line[0] for line in texts])
#%%
print('Preparing fields...')

dep_options = ['MICU','ONCO','ACOVE','BLUE','BLUMGART','CARD','CCU','CME','CORONARY','CSU','GENERAL','MED','NEO','NEURO','OBST','ORTHO','OTOLA','ICU','PLASTIC','PSY','SURGERY','THORAC','TRASNPLANT','TRAUMA','VSU','CT']

for i in range(len(dep)):
    dep[i].replace('DOCTOR','MEDDOC')
    count = 0
    for op in dep_options:
        if op in dep[i]:
            dep[i] = op
            break
        else: count += 1
    if count == 24: dep[i] = 'NONE'
    
#%%
for i in range(len(res_clinic)):
    if len(res_clinic[i]) > 25: res_clinic[i] = res_clinic[i][:25]
    if isinstance(diag_alta[i],str): diag_alta[i] = [diag_alta[i]]
    if len(diag_alta[i]) > 9: diag_alta[i] = diag_alta[i][:9]
    mot_int[i] = '873heGKe7I ' + mot_int[i] + ' 873heGKe7I'
    for j in range(len(res_clinic[i])):
        res_clinic[i][j] = '873heGKe7I ' + res_clinic[i][j] + ' 873heGKe7I'
        if res_clinic[i] == '873heGKe7I  873heGKe7I' or res_clinic[i] == '873heGKe7I   873heGKe7I': res_clinic[i] = ''
    for j in range(len(diag_alta[i])):
        diag_alta[i][j] = '873heGKe7I ' + diag_alta[i][j] + ' 873heGKe7I'
        if diag_alta[i] == '873heGKe7I  873heGKe7I' or diag_alta[i] == '873heGKe7I   873heGKe7I': diag_alta[i] = ''
    while len(res_clinic[i]) < 25:
        res_clinic[i].append('')
    while len(diag_alta[i]) < 9:
        diag_alta[i].append('')

#%%
print('Preparing discharge summaries file format...')

texts_aux = list(res_clinic)
for i in range(len(res_clinic)):
    if labels_cid[i] != [''] and len(labels_cid[i]) != 0:
        texts_aux[i] = labels_cid[i][0] + ' <> ' + diag_alta[i][0] + ' <> ' + diag_alta[i][1] + ' <> ' + diag_alta[i][2] + ' <> ' + diag_alta[i][3] + ' <> ' + diag_alta[i][4] + ' <> ' + diag_alta[i][5] + ' <> ' + diag_alta[i][6] + ' <> ' + diag_alta[i][7] + ' <> ' + diag_alta[i][8] + ' <> ' + mot_int[i] + ' <> ' + res_clinic[i][0] + ' <> ' + res_clinic[i][1] + ' <> ' + res_clinic[i][2] + ' <> ' + res_clinic[i][3] + ' <> ' + res_clinic[i][4] + ' <> ' + res_clinic[i][5] + ' <> ' + res_clinic[i][6] + ' <> ' + res_clinic[i][7] + ' <> ' + res_clinic[i][8] + ' <> ' + res_clinic[i][9] + ' <> ' + res_clinic[i][10] + ' <> ' + res_clinic[i][11] + ' <> ' + res_clinic[i][12] + ' <> ' + res_clinic[i][13] + ' <> ' + res_clinic[i][14] + ' <> ' +  res_clinic[i][15] + ' <> ' +  res_clinic[i][16] + ' <> ' + res_clinic[i][17] + ' <> ' + res_clinic[i][18] + ' <> ' + res_clinic[i][19] + ' <> ' + res_clinic[i][20] + ' <> ' + res_clinic[i][21] + ' <> ' + res_clinic[i][22] + ' <> ' + res_clinic[i][23] + ' <> ' + res_clinic[i][24] + ' <> ' + dep[i] + ' <> ' + str(labels_cid[i])
        texts_aux[i] = texts_aux[i] + '\r'

print('Writing output...')

f = open('dataset_input_mimic.txt','w')

for _list in texts_aux:
    if isinstance(_list,str): f.write(_list)
    
f.close()
