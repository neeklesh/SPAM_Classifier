#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SPAM Message classifier


# In[1]:


import pandas as pd
import numpy as np
import re
import nltk


# In[4]:


nltk.download('stopwords')


# In[3]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[5]:


df = pd.read_csv('D:\Python\Spam.csv',names=["label", "message"])


# In[6]:


df.head()


# # Data Cleaning

# In[7]:


ls = WordNetLemmatizer()


# In[8]:


ls


# In[9]:


nltk.download('wordnet')


# In[10]:


nltk.download('omw-1.4')


# In[11]:


corpus = []
for i in range(len(df)):
    review =re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [ls.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) #we join all the words in review
    corpus.append(review)


# In[12]:


corpus


# # #creating bag of words

# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
Z = cv.fit_transform(corpus).toarray()


# In[62]:


Z.shape


# In[16]:


#splitting the data into dependent and independent variable
y = pd.get_dummies(df['label'])
y 


# In[17]:


y = y.iloc[:,1].values


# In[18]:


y


# In[20]:


#converting the data into training and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size = 0.20, random_state = 0)


# In[26]:


y_train.shape


# In[27]:


#importing Naive Baeys classifier for NLP
from sklearn.naive_bayes import MultinomialNB
Spam_detect_modal = MultinomialNB().fit(X_train,y_train)


# In[28]:


Spam_detect_modal


# In[29]:


y_pred = Spam_detect_modal.predict(X_test)
y_pred


# In[30]:


y_test


# In[32]:


print('Predicted vale', y_pred, 'test Value', y_test)


# # Testing the results

# In[34]:


#We can use accuacy score and confusion matrix
from sklearn.metrics import confusion_matrix


# In[35]:


con = confusion_matrix(y_test, y_pred)


# In[36]:


con


# In[37]:


from sklearn.metrics import accuracy_score


# In[40]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[43]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[42]:


report


# # Now lets use Stemming

# In[45]:


ps = PorterStemmer()
ps


# In[63]:


corp = []
for i in range(len(df)):
    rev = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    corp.append(rev)


# In[64]:


corp


# # Now we create bag of words

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer


# In[66]:


vector = CountVectorizer()
vector


# In[67]:


x = vector.fit_transform(corp).toarray()
x.shape


# In[60]:


y.shape


# # Splitting the data in Training and Testing

# In[70]:


#converting the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state=0)


# In[96]:


y_test.shape


# In[71]:


Spam_detect_modal = MultinomialNB().fit(X_train,y_train)


# In[73]:


#predicting after splitting and creating the naive baeys modal
y_pred_vec = Spam_detect_modal.predict(X_test)


# # Testing the result

# In[74]:


#we can use confusion matrix, accuracyscore and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[75]:


CM = confusion_matrix(y_test, y_pred_vec)


# In[76]:


CM


# In[77]:


Acc_score_vec = accuracy_score(y_test, y_pred_vec)
Acc_score_vec


# In[78]:


classification_report_vec = print(classification_report(y_test,y_pred_vec))


# # RandomForest

# In[101]:


#lets use random forrest
from sklearn.ensemble import RandomForestClassifier


# In[80]:


rf = RandomForestClassifier()


# In[81]:


rf


# In[82]:


x = rf.fit(X_train, y_train)


# In[83]:


x


# # predicting using Randomforrest

# In[84]:


y_pred_random = x.predict(X_test)


# In[85]:


y_pred_random


# In[86]:


#Now testing the accuracy
CM_Ran = confusion_matrix(y_test,y_pred_random)


# In[87]:


CM_Ran


# In[89]:


Accuracy_random = accuracy_score(y_test,y_pred_random)
Accuracy_random


# # Decision Tree

# In[99]:


#converting the data into training and testing
from sklearn.model_selection import train_test_split


# In[107]:


X_train.shape


# In[ ]:





# In[108]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=7, random_state = 48) 
dt.fit(X_train,y_train)

y_pred_decision = dt.predict(X_test)


# In[110]:


#testing accuracy of the prediction
CM_DS = confusion_matrix(y_test,y_pred_decision)
Accuracy_DS = accuracy_score(y_test,y_pred_decision)

print ('The confusion matrix of Decision Tree is:', CM_DS)
print ('The Accuracy of Decision Tree is:', Accuracy_DS)
Classification_ds = print(classification_report(y_test,y_pred_decision))


# In[95]:


y


# In[114]:


print('Accuracy Score using Naive Bayes classifier and TF-IDF vectorizer & Lemmatisation is:', 96.5 )
print('Accuracy Score using Naive Bayes classifier and TF-IDF vectorizer & Stemming is:', 97.04 )
print('Accuracy Score using Random Forest classifier and TF-IDF vectorizer & Stemming is:', 97.39 )
print('Accuracy Score using Random Decison tree and TF-IDF vectorizer & Stemming is:', 93.99 )


# In[ ]:




