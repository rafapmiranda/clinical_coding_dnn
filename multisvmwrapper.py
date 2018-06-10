#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sklearn
import pickle
import numpy as np
from keras.models import Model
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset

# Wraper over the Label Powerset multi-label classifier from skmultilearn with a SVM base classifier from sklearn
class SVMWrapper(Model):

    def __init__(self, C=1.0, use_idf=True, filename=None, **kwargs):
        self.svmmodel = LabelPowerset(LinearSVC( C=C , random_state=0 ))
        self.vect1 = TfidfVectorizer(norm=None, use_idf=use_idf, min_df=0.0)
        self.output_dim = 0
        if filename is not None: self.load(filename)

    def build_representation( self, x, fit=False ):
        auxX = [ ' '.join( [ 'w_'+str(token) for token in instance if token != 0 ] ) for instance in x ]
        if fit : return self.vect1.fit_transform(auxX)
        else : return self.vect1.transform(auxX)

    def fit( self, x, y, validation_data=None):
        auxX = self.build_representation(x,fit=True)
        auxY = y
        self.svmmodel.fit( auxX , auxY )
        self.output_dim = auxY.shape[1]
        if validation_data is None: return None
        res = self.evaluate( validation_data[0] , validation_data[1] )
        print("Precision in validation data =",res)
        return None
		
    def predict(self, x):
        auxX = self.build_representation(x,fit=False)
        auxY = self.svmmodel.predict(auxX)
        if auxY.shape[1] < self.output_dim:
            npad = ((0, 0), (0, self.output_dim-auxY.shape[1]))
            auxY = np.pad(auxY, pad_width=npad, mode='constant', constant_values=0)
        return [ auxY, [], [] ]
        
    def evaluate(self, x, y):
        auxX = self.build_representation(x,fit=False)
        auxY = y
        auxY = np.array([ np.argmax(i) for i in auxY ])
        return sklearn.metrics.precision_score(y_true=auxY,y_pred=self.svmmodel.predict(auxX), average='weighted')
	
    def save(self, filename):
        f = open(filename, "wb")
        pickle.dump(self.svmmodel, f)
        pickle.dump(self.vect1, f)
        pickle.dump(self.output_dim, f)
        f.close()
	
    def load(self, filename): 
        f = open(filename, "rb")
        self.svmmodel = pickle.load(f)
        self.vect1 = pickle.load(f)
        self.output_dim = pickle.load(f)
        f.close()
