import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import metrics

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif as  MIC

def readData():
    data = pd.read_csv('preprocess_train.csv',encoding='utf-8')
    y = data['label']
    x = data.iloc[:,0:-1]
    #查看类别标签分布情况
    t,vec = np.unique(y, return_counts=True)
    #print(t)
    #print(vec)
    
    #有三个特征是只有一个值的，都是0
    #剔除只有一个值的特征
    for col in x.columns:
        if(x[col].nunique()<=1):
            print(col)
            x.drop(col,axis=1,inplace=True)
    
    return x,y
    
#进行缺失值填补
def KNNimpute(x):
    #print(x)
    imputer = KNNImputer(n_neighbors=10,weights="uniform")
    x = imputer.fit_transform(x)
    x = pd.DataFrame(x)
    #print(x)
    return x
    
#进行互信息筛选
def MICompute(train,y):
    result=MIC(train,y)
    #new_data = SelectKBest(MIC, k=20).fit_transform(train,y)
    feature = SelectKBest(MIC, k=10).fit(train,y).get_support(indices=True)
    print(feature)
    #new_data = pd.DataFrame(new_data)
    return feature

#标准化
def standardization(train):
    scaler = preprocessing.StandardScaler().fit(train)
    X_scaled = scaler.transform(train)
    X_scaled = pd.DataFrame(X_scaled)
    return X_scaled

#分类
def histGradientClassifier(train,y):
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(train):
        train_X, train_y = train.iloc[train_index], y.iloc[train_index]
        test_X, test_y = train.iloc[test_index], y.iloc[test_index]
        clf = HistGradientBoostingClassifier()
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        print(metrics.f1_score(test_y,pred_y,average='macro'))
        
        
def test():
    x,y = readData()
    train = KNNimpute(x)
    #train = standardization(train)
    feature = MICompute(train,y)
    train = train.iloc[:,feature]
    print(train)
    histGradientClassifier(train,y)
    #原始数据集
    #histGradientClassifier(x,y)
    
test()