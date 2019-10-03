
import jieba
import pandas as pd
import numpy as np
import time
import datetime
# import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import matplotlib.pyplot as plt
# from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier



##读入训练集
train=pd.read_table("apptype_train.dat",sep=' ',header=None)
train=pd.DataFrame(train[0].apply(lambda x:x.split('\t')).tolist(),columns=['id','label','des'])
train['label1']=train['label'].apply(lambda x:x.split('|')[0])
train['label2']=train['label'].apply(lambda x:x.split('|')[1] if '|' in x else '0')

print("读入训练集完成")

##读入测试集
test= pd.read_csv("app_desc.dat",header=None,encoding='utf8',delimiter=' ')
test=pd.DataFrame(test[0].apply(lambda x:x.split('\t')).tolist(),columns=['id','des'])

print("读入测试集完成")

##remove the samples that are less than 5

train=train[~train.label1.isin(['140110','140805','140105'])]
##remove words that is less than 10 except if it contains "实时美颜"

removeSample=train[train.des.map(len)<10]
removeSample=removeSample[~removeSample.des.str.contains('实时美颜')]
train=train[~train.index.isin(removeSample.index)]


##对描述进行分词
def readStopWords():
    fileStream=open('./stopwords-master/中文停用词表.txt',"r", encoding='utf8', errors='ignore')
    content=fileStream.read()
    return content

def removeStopWords(wordList):
    content=readStopWords()
    outstr=' '
    for word in wordList:
        new_word= filter(str.isalpha, word)
        new_word=''.join(list(new_word))
        if new_word not in content:
            if new_word != '\t' and '\n':
                outstr=outstr+' '+new_word
                continue
    return outstr

'''
@param 输入的app描述 (String)
@return 去处+分词后的描述 
'''
def splitWordS(des):
    res=jieba.cut(des)
    listcontent=''
    for word in res:
        listcontent += word
        listcontent += " "
    wordList=listcontent.split(' ')
    return removeStopWords(wordList)


train.des=train.des.apply(lambda x:splitWordS(x))

'''
对标签进行编码
'''

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train.label1);
train.label1=le.transform(train.label1)


le2 = preprocessing.LabelEncoder()
le2.fit(train.label2);
train.label2=le2.transform(train.label2)

print("编码完成")


'''
切割数据
'''
x_train, x_test, y_train, y_test = train_test_split(train.des,train.label1, test_size=0.25)
x_train2, x_test2, y_train2, y_test2 = train_test_split(train.des,train.label2, test_size=0.25)



'''
交叉验证分类
'''

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
import numpy as np

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(shuffle=False, n_splits=K, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print (np.mean(scores))


nbc_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])


nbc_2 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

nbc_3= Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8,use_idf=1,smooth_idf=1, sublinear_tf=1)),
    ('clf', OneVsRestClassifier(MultinomialNB())),
])

nbc_4= Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8,use_idf=1,smooth_idf=1, sublinear_tf=1)),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01))),
])

nbc_5 = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8)),
    ('clf', MultinomialNB()),
])
nbc_6=Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8)),
    ('clf', MultinomialNB(alpha=0.01)),
])


# nbcs=[nbc_1,nbc_2,nbc_3,nbc_4,nbc_5,nbc_6]
# for nbc in nbcs:
#     evaluate_cross_validation(nbc, x_train, y_train, 5)
'''
综上，nbc_4被认为效果最好
'''

test['des']=test['des'].apply(lambda x:splitWordS(x))
print('测试集为',test['des'])

nbc_4.fit(x_train, y_train)
y_predict = nbc_4.predict(test['des'])
y_predict=le.inverse_transform(list(y_predict))
results=pd.DataFrame(y_predict)
results.rename(columns={results.columns[0]: "label1"}, inplace=True)
print("测试集第一列为")
print(results)


nbc_4.fit(x_train2, y_train2)
y_predict2 = nbc_4.predict(test['des'])
y_predict2=le.inverse_transform(list(y_predict2))
results['label2']=pd.DataFrame(y_predict2)
print("测试集第二列为")
print(results)



pd.concat([test[['id']],results[['label1','label2']]],axis=1).to_csv('submit2.csv',index=None,encoding='utf8')
print("已经生成结果")


# pd.concat([test[['id']],results[['label1','label2']]],axis=1).to_csv('submit.csv',index=None,encoding='utf8')






