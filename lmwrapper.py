#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sklearn
import pickle
import numpy as np
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import TfidfVectorizer

# Wraper over the MultinomialNB classifier from sklearn
class LMWrapper(Model):

    def __init__(self, C=1.0, use_idf=False, filename=None, **kwargs):
        self.lm = MultinomialNB()
        self.vect1 = TfidfVectorizer(norm=None, use_idf=use_idf, min_df=0.0, ngram_range=(1, 1))
        self.selector = sklearn.feature_selection.SelectKBest(k='all')
        self.output_dim = 0
        if filename is not None: self.load(filename)

    def build_representation( self, x, y=None, fit=False ):
        auxX = [ ' \n '.join( [ ' '.join( [ 'w_'+str(token) for token in field if token != 0 ] ) for field in instance ] ) for instance in x ]
        if fit: self.vect1.fit(auxX)
        auxX = self.vect1.transform(auxX)
        if fit: self.selector.fit(auxX,np.array([ np.argmax(i) for i in y ]))
        auxX = self.selector.transform(auxX)
        return auxX.todense()
        
    def fit( self, x, y, validation_data=None):
        auxY = y
        print('Build representation...')
        auxX = self.build_representation(x,auxY,fit=True)
        #self.lm.fit( auxX , np.array([ np.argmax(i) for i in auxY ]) )
        print('auxX shape:',auxX.shape)
        print('Fit model...')
        self.lm.fit( auxX , np.array([ np.argmax(i) for i in auxY ]) )
        self.output_dim = auxY.shape[1]
        if validation_data is None: return None
        res = self.evaluate( validation_data[0] , validation_data[1] )
        print("Accuracy in validation data =",res)
        return None
		
    def predict(self, x):
        auxX = self.build_representation(x,fit=False)
        print('Predicting baseline...')
        auxY = self.lm.predict(auxX)
        #auxY = to_categorical(auxY)
        if auxY.shape[1] < self.output_dim:
            npad = ((0, 0), (0, self.output_dim-auxY.shape[1]))
            auxY = np.pad(auxY, pad_width=npad, mode='constant', constant_values=0)
        return [ auxY, [], [] ]
	
    def predict_prob(self, x):
        auxX = self.build_representation(x,fit=False)
        print('Predicting baseline...')
        auxY = self.lm.predict_proba(auxX)
        if auxY.shape[1] < self.output_dim:
            npad = ((0, 0), (0, self.output_dim-auxY.shape[1]))
            auxY = np.pad(auxY, pad_width=npad, mode='constant', constant_values=0)
        return [ auxY, [], [] ]
        
    def evaluate(self, x, y):
        auxX = self.build_representation(x,fit=False)
        auxY = y
        auxY = np.array([ np.argmax(i) for i in auxY ])
        return sklearn.metrics.accuracy_score(y_true=auxY,y_pred=self.lm.predict(auxX))
	
    def save(self, filename):
        f = open(filename, "wb")
        pickle.dump(self.lm, f, protocol=4)
        pickle.dump(self.vect1, f, protocol=4)
        pickle.dump(self.selector, f, protocol=4)
        pickle.dump(self.output_dim, f, protocol=4)
        f.close()
	
    def load(self, filename): 
        f = open(filename, "rb")
        self.lm = pickle.load(f)
        self.vect1 = pickle.load(f)
        self.selector = pickle.load(f)
        self.output_dim = pickle.load(f)
        f.close()
