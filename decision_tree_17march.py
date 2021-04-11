#!/usr/bin/env python
# coding: utf-8

# # decision_tree_17march

# In[1]:


#import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[2]:


#plot th DT
from sklearn import tree
from IPython.display import Image
from subprocess import check_call
#to make the image and dowload in pdf


# In[3]:


import numpy as np
import seaborn as sns  #can be used for multicolinearity #it is a ploting tool
import matplotlib.pyplot as plt


# In[4]:


#RFE (recusive feature elimination) feature selection technique
from sklearn.feature_selection import RFE


# In[5]:


import pandas as pd


# In[6]:


#read the data 
path ="C:/Users/mayur/Desktop/datascience DELL/pythonstorage/dataset_ML/ecoli.csv"
ecoli=pd.read_csv(path)


# In[7]:


ecoli.head()


# In[8]:


ecoli.shape


# In[9]:


#drop the column 'sequence name'
ecoli = ecoli.drop('sequence_name',axis=1)


# In[10]:


ecoli.head()


# In[11]:


#check for singularities 
ecoli.lip.value_counts()


# In[12]:


326/len(ecoli)  #case of singularity


# In[13]:


ecoli.chg.value_counts()


# In[14]:


335/len(ecoli)


# In[15]:


#check the distribution of Y-classes
ecoli.lsp.value_counts()
#it will be more biased towards cp ->42%


# In[16]:


143/len(ecoli)


# In[17]:


#shuffle the data since Y is grouped
ecoli = ecoli.sample(frac=1) #fraction with 100%


# In[18]:


ecoli.head()


# In[ ]:


#perform EDA


# In[19]:


ecoli.isnull().sum()


# In[20]:


#0 check 
ecoli[ecoli==0].count()


# In[21]:


#split the data into train and test
trainx,testx,trainy,testy = train_test_split(ecoli.drop('lsp',axis=1),ecoli.lsp,test_size=0.2)


# In[22]:


print("trainx={},trainy={},testx={},testy ={}".format(trainx.shape,trainy.shape,testx.shape,testy.shape))


# In[ ]:


#there 2 DT models
#1) Entropy model
#2) Gini model 
#ccp_alpha --> cost parameter 


# In[23]:


#Entropy model - without HPT
m1=DecisionTreeClassifier(criterion='entropy').fit(trainx,trainy)


# In[24]:


print(m1)


# In[25]:


help(DecisionTreeClassifier)


# In[26]:


#plot the decision tree 
features = list(ecoli.columns)
features.remove('lsp')
classes =  ecoli.lsp.unique()


# In[27]:


#create the tree
tree.export_graphviz(m1,'m1tree1.dot',filled=True,rounded=True,feature_names=features,class_names=classes)


# In[28]:


#convert dot to image file
check_call(['dot','-Tpng','m1tree1.dot','-o','m1tree1.png'])


# In[29]:


Image(filename='m1tree1.png')


# In[30]:


#predictions 
p1=m1.predict(testx)


# In[31]:


#confusion matrix/ classification report/ accuracy score
accuracy_score(testy,p1)


# In[32]:


#confusion matrix
confusion_matrix(testy,p1)


# In[33]:


df= pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df.actual,df.predicted,margins=True)


# In[34]:


print(classification_report(testy,p1))


# In[35]:


24/29


# In[36]:


len(ecoli.lsp.unique())


# In[ ]:





# In[37]:


#important features
m1.feature_importances_   #works on train data


# In[38]:


#create dataframe to store the features name and their scores
#higher score = high significance
impf= pd.DataFrame({'features':trainx.columns,'score':m1.feature_importances_})


# In[39]:


#sort the data by scores in decreasing order
impf.sort_values('score',ascending=False,inplace=True)


# In[40]:


#plot the significant features
sns.barplot(x=impf.score,y=impf.features)
plt.title('Decision Tree - Significant Features')
plt.xlabel('Score')
plt.ylabel('Features')


# In[41]:


#Decision Tree pruning 
dt_path = m1.cost_complexity_pruning_path(trainx,trainy)
dt_path


# In[42]:


#cost compexity parameter values
ccp_alphas = dt_path.ccp_alphas 
ccp_alphas


# In[43]:


#find the best ccp_alpha value
results = []
for cp in ccp_alphas:
    m = DecisionTreeClassifier(ccp_alpha = cp).fit(trainx,trainy)
    results.append(m)
results    


# In[44]:


#calculate the accuracy scores for train and test data
trg_score = [r.score(trainx,trainy)for r in results]
test_score = [r.score(testx,testy)for r in results]


# In[45]:


#plot the scores
fig,ax = plt.subplots()
ax.plot(ccp_alphas,trg_score,marker='o',label='train',drawstyle = 'steps-post')
ax.plot(ccp_alphas,test_score,marker='o',label='test',drawstyle='steps-post')
ax.set_xlabel("CCP alpha")
ax.set_ylabel("Accuracy")
ax.set_title("CCP Alpha vs Accuracy")
ax.legend()
#these are the accuracy of train and test distance(alpha) between them should be less to 


# In[46]:


#based on the graph , the best ccp_alpha=0.023
#build model with this ccp_alpha value
#for better results, experiment with the ccp_alpha values
m1_1=DecisionTreeClassifier(criterion='entropy',ccp_alpha = 0.023).fit(trainx,trainy)
p1_1 = m1_1.predict(testx)
p1_1


# In[47]:


p1_1


# In[48]:


df1_1 = pd.DataFrame({'actual':testy,'predicted':p1_1})


# In[49]:


pd.crosstab(df1_1.actual,df1_1.predicted,margins=True)
print(classification_report(testy,p1_1))


# In[50]:


# 2 Entropy model with hyperparameter
m2 = DecisionTreeClassifier(criterion='entropy',
                            max_depth=2,
                            min_samples_leaf=3).fit(trainx,trainy)


# In[51]:


p2=m2.predict(testx)
p2


# In[52]:


df2 = pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(testy,p2))


# In[ ]:





# In[ ]:


# feature  selection 

#next models m3 and m4
#criterion ="gini"
#follow the same steps as above


# In[53]:


#feature selection -Method 2  -- >#RFE (recursive feature Elimination)
cols = list(testx.columns)   #works on test data


# In[54]:



m1


# In[61]:


#specify the number of significant features you want from the model 
features = 3
rfe= RFE(m1,features).fit(testx,testy)
support = rfe.support_
ranking = rfe.ranking_
#store the results in dataframes
df_rfe = pd.DataFrame({'feature':cols,'support':support,'rank':ranking})
#sort the dataframe by rank
df_rfe.sort_values('rank',ascending =True,inplace=True)
print(df_rfe)


# In[56]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




