# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 09:52:14 2021

@author: saz2n
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:31:05 2021

@author: saz2n
"""
import numpy as np
import requests
import json
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#from preprocessor import Preprocess
import flask
from flask import Flask,request

app = Flask(__name__)

class NLP:
    
    def fitVectorizer(self,xTrain):
        vect = TfidfVectorizer()
        vect.fit_transform(xTrain)
        return vect
    
    def transform(self,vect,data):
        return vect.transform(data)
    
class ML:
    
    def __init__(self):
        self.__clf = None
        self.__vect = None
    
    def dataLoadFromApi(self):
        url = "http://localhost:80/fetch"
        payload={}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        return data

    def dataLoadFromExcel(self,path,sep):
        data = pd.read_csv(path,sep=sep)
        return data

    def convertToDf(self,dataDict):
        classNames = dataDict.keys()
        dataList = []
        classList = []
        for className in classNames:
            dataList += dataDict[className]
            classList += [className]*len(dataDict[className])
        df = pd.DataFrame({'articles':dataList,'label':classList})
        return df

    def splitData(self,df,xColName,yColName,test_size=0.2):
        xTrain,xTest,yTrain,yTest = train_test_split(df[xColName],df[yColName],test_size=test_size)
        return xTrain,xTest,yTrain,yTest
    
    def trainModel(self,tfidfTrain,yTrain):
        clf = RandomForestClassifier()
        clf.fit(tfidfTrain,yTrain)
        return clf
    
    def predict(self,xTest):
        #preprocessObj = Preprocess()
        #xTestProcessed = [preprocessObj.removeSpecialChar(text) for text in xTest]
        nlpObj = NLP()
        tfidfMatrix = nlpObj.transform(self.__vect,xTest)
        return self.__clf.predict(tfidfMatrix)
    
    def main(self):
        path = 'data/train.tsv'
        df = self.dataLoadFromExcel(path,'\t')
        #df = self.convertToDf(dataDict)
        #preprocessObj = Preprocess()
        #df['articles'] = df.articles.apply(lambda x: preprocessObj.removeSpecialChar(x))
        xTrain,xTest,yTrain,yTest = self.splitData(df,'Phrase','Sentiment')
        nlpObj = NLP()
        self.__vect = nlpObj.fitVectorizer(xTrain)
        tfidfMatrix = nlpObj.transform(self.__vect,xTrain)
        self.__clf = self.trainModel(tfidfMatrix,yTrain)
        
mlObj = ML()
mlObj.main()

@app.route('/predict_class',methods=['POST'])
def givePred():
       dataDict = json.loads(request.data.decode())      # request.data is encoded,this is json..convert to dict 
       preds = mlObj.predict(dataDict['Phrase'])
       return json.dumps({'class':list(preds)})
       
app.run('localhost',5004)
        
        
        
        
        
        
        