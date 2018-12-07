# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:11:49 2018

@author: Rafael
"""
#%%
from __future__ import print_function
import numpy as np
import codecs

np.random.seed(1337) # for reproducibility

def ex_based_acc(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. example-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        #print('\nInstance no. %s' % (i+1))
        #print('set_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        #print(set_true.intersection(set_pred))
        #print(set_true.union(set_pred))
        tmp_a = len(set_true.intersection(set_pred))/\
                float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def ex_based_precision(y_true, y_pred):
    '''
    Compute the example-based precision for the multi-label case
    '''
    p_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        tmp_a = len(set_true.intersection(set_pred))/\
                float( len(set_pred) )
        p_list.append(tmp_a)
    return np.mean(p_list)

def ex_based_recall(y_true, y_pred):
    '''
    Compute the example-based recall for the multi-label case
    '''
    r_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = len(set_true.intersection(set_pred))/\
                float( len(set_true) )
        #print('tmp_a: {0}'.format(tmp_a))
        r_list.append(tmp_a)
    return np.mean(r_list)

def ex_based_f1(y_true, y_pred):
    '''
    Compute the example-based f1 measure for the multi-label case
    '''
    f_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        tmp_a = 2*len(set_true.intersection(set_pred))/\
                float( len(set_true) + len(set_pred) )
        f_list.append(tmp_a)
    return np.mean(f_list)

def multi_hamming_loss(y_true, y_pred):
    '''
    Compute the hamming loss for the multi-label case
    '''
    label_size = 0
    for i in range(y_true.shape[0]):
        label_size += len(y_true[i])
        label_size += len(y_pred[i])
        
    h_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        tmp_a = len(set_true.symmetric_difference(set_pred))/\
                float( label_size )
        h_list.append(tmp_a)
    return np.mean(h_list)

def exact_match(y_true, y_pred):
    '''
    Compute the exact match measure for the multi-label case
    '''
    m_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = set( y_pred[i] )
        if set_true == set_pred:
            m_list.append(1)
        else:
            m_list.append(0)
    return np.mean(m_list)

def one_error(y_true, y_pred):
    '''
    Compute the one error measure for the multi-label case
    '''
    m_list = []
    for i in range(y_true.shape[0]):
        set_true = set( y_true[i] )
        set_pred = y_pred[i][0]
        if set_pred not in set_true:
            m_list.append(1)
        else:
            m_list.append(0)
    return np.mean(m_list)

def top_labels(max_rank, y_true, y_pred):
    '''
    Compute the score associated to true labels being between the top-max_rank predicted labels
    '''    
    m_list = []
    for i in range(y_true.shape[0]):
        set_pred = y_pred[i]
        if y_true[i] in set_pred:
            m_list.append(1)
        else:
            m_list.append(0)
    return np.mean(m_list)
