#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on youtube and tiktok appstore, and maybe pytrend 

# In[1]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from app_store_scraper import AppStore

from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction import _stop_words
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import matplotlib.pylab as plt
from dmba import printTermDocumentMatrix, classificationSummary, liftChart
import re

# sentiment libs
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# # some duplicate libraries are here
# import pandas as pd
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# from app_store_scraper import AppStore
# 
# 
# from zipfile import ZipFile
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction import _stop_words
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
# import nltk
# from nltk import word_tokenize
# from nltk.stem.snowball import EnglishStemmer
# import matplotlib.pylab as plt
# from dmba import printTermDocumentMatrix, classificationSummary, liftChart
# from nltk.corpus import stopwords
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# 
# 
# import nltk
# #  nltk.download('punkt')
# #  nltk.download('stopwords')
# 
# #sentiment libs
# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# # Saved the scraped data into excel file. Don't run code in this session, start from the exploration part , each app scraped 1000 records
# 

# In[2]:


" tiktok = AppStore(country='us', app_name='tiktok', app_id = '835599320') "

" tiktok.review(how_many=1000) "

#delete the quotes to scrap 

# this is tiktok review


# In[3]:


# transform the scrapped data

"""" tiktokdf = pd.DataFrame(np.array(tiktok.reviews),columns=['review'])
tiktokdf2 = tiktokdf.join(pd.DataFrame(tiktokdf.pop('review').tolist())) """"


# In[17]:


# writer_123 = pd.ExcelWriter('C:/Users/qifan/OneDrive/Documents/MDA 720/Tiktokdf.xlsx', engine='xlsxwriter')

# tiktokdf2.to_excel(writer_123, sheet_name='emails', index=False)

# writer_123.save()



# #download as excel


# In[ ]:





# In[ ]:


#  Youtube data


# In[18]:


"""""   youtube = AppStore(country='us', app_name='YouTube', app_id = '544007664')

youtube.review(how_many=1000)     """""

#delete the quotes to scrap 

# this is youtube review


# In[21]:


""""  youtubedf = pd.DataFrame(np.array(youtube.reviews),columns=['review'])
      youtubedf2 = youtubedf.join(pd.DataFrame(youtubedf.pop('review').tolist())) """"


# In[24]:


# writer_1234 = pd.ExcelWriter('C:/Users/qifan/OneDrive/Documents/MDA 720/Youtubedf.xlsx', engine='xlsxwriter')

# youtubedf2.to_excel(writer_1234, sheet_name='youtube_df', index=False)

# writer_1234.save()



# download as excel


# In[ ]:





# In[ ]:





# # Data exploration tiktok

# In[2]:


# read local excel file instead of using scraped file everytime opening this file


# In[3]:


tiktokdf2= pd.read_excel("C:/Users/qifan/OneDrive/Documents/MDA 720/Tiktokdf.xlsx")
youtubedf2=pd.read_excel("C:/Users/qifan/OneDrive/Documents/MDA 720/Youtubedf.xlsx")


# In[4]:


tiktokdf2.head()


# In[5]:


tiktokdf2.dtypes


# In[6]:


len(tiktokdf2)


# In[7]:


import matplotlib.pyplot as plt

counts = tiktokdf2['rating'].value_counts(ascending=False)

plt.bar(counts.index, counts.values)

plt.xlabel('Rating')
plt.ylabel('Frequency')

plt.title('Distribution of TikTok Ratings')

plt.show()


# In[8]:


tiktokdf2['rating'].value_counts(ascending=False)


# # data exploration youtube

# In[9]:


youtubedf2.head()


# In[10]:


youtubedf2.dtypes
len(youtubedf2)


# In[11]:


counts_1 = youtubedf2['rating'].value_counts(ascending=False)

plt.bar(counts_1.index, counts_1.values)

plt.xlabel('Rating')
plt.ylabel('Frequency')

plt.title('Distribution of Youtube Ratings')

plt.show()


# In[12]:


youtubedf2['rating'].value_counts(ascending=False)


# In[13]:


# rating comparison 


# In[14]:


#
print('tiktok rating median is',np.median(tiktokdf2['rating']))
print('youtube rating median is',np.median(youtubedf2['rating']))


# In[15]:


tiktokdf2['rating'].value_counts(ascending=False)


# In[16]:


youtubedf2['rating'].value_counts(ascending=False)


# In[17]:


# converting 123 to 0 and 4,5 to 1 on sentiment , to two category


# In[18]:


#
# Assign 0 for reviews <= 3 (negative sentiment)
tiktokdf2.loc[tiktokdf2['rating'] <= 3, 'sentiment'] = 0

# Assign 1 for reviews > 3 (positive sentiment)
tiktokdf2.loc[tiktokdf2['rating'] > 3, 'sentiment'] = 1


# In[19]:


#convert float sentiment to int
tiktokdf2['sentiment'] = tiktokdf2['sentiment'].astype(int)


# In[20]:


#
# Assign 0 for reviews <= 3 (negative sentiment)
youtubedf2.loc[youtubedf2['rating'] <= 3, 'sentiment'] = 0

# Assign 1 for reviews > 3 (positive sentiment)
youtubedf2.loc[youtubedf2['rating'] > 3, 'sentiment'] = 1


# In[21]:


youtubedf2['sentiment'] =youtubedf2['sentiment'].astype(int)


# In[22]:


tiktokdf2['sentiment'].value_counts()


# In[23]:


youtubedf2['sentiment'].value_counts()


# In[24]:


youtubedf2


# In[25]:


# perform stop words to make new column clean_review to both tiktok and youtube

stp_words = stopwords.words('english')

def clean_review(review):
  clean_review = " ".join(word for word in review.split() if word not in stp_words)
  return clean_review

tiktokdf2['clean_review'] = tiktokdf2['review'].apply(clean_review)


# In[26]:


# cleaning the review from youtube
stp_words = stopwords.words('english')

def clean_review(review):
  clean_review = " ".join(word for word in review.split() if word not in stp_words)
  return clean_review

youtubedf2['clean_review'] = youtubedf2['review'].apply(clean_review)


# In[71]:


from wordcloud import WordCloud

# tiktok 1,2,3 star rating wordcloud
consolidated=' '.join(word for word in tiktokdf2['review'][tiktokdf2['sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110, colormap='Paired')
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[70]:


# # most frequent 30 words 1,2,3 star rating tiktok
# freq_dict = wordCloud.process_text(consolidated)
# sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
# for i, (word, freq) in enumerate(sorted_dict[:40]):
#     print(f"{i+1}. {word}: {freq}")


# In[72]:


# most frequen 40 words rating 1,2,3
freq_dict = wordCloud.process_text(consolidated)
sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

for i in range(0, min(len(sorted_dict), 40), 20):
    batch = sorted_dict[i:i+20]
    for j, (word, freq) in enumerate(batch):
        print(f"{i+j+1}. {word}: {freq}")
    print()


# In[73]:


# word-cloud 4,5 star, tiktok
consolidated=' '.join(word for word in tiktokdf2['review'][tiktokdf2['sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110, colormap='Paired')
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[74]:


# most frequent 40 words 4,5 star rating tiktok
freq_dict = wordCloud.process_text(consolidated)
sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

for i in range(0, min(len(sorted_dict), 40), 20):
    batch = sorted_dict[i:i+20]
    for j, (word, freq) in enumerate(batch):
        print(f"{i+j+1}. {word}: {freq}")
    print()


# In[76]:


#  YOUTUBE 

# youtube 1,2,3 star rating wordcloud
consolidated=' '.join(word for word in youtubedf2['review'][youtubedf2['sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110, colormap='Paired')
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[77]:


# most frequent 40 words 1,2,3 star rating youtube
freq_dict = wordCloud.process_text(consolidated)
sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

for i in range(0, min(len(sorted_dict), 40), 20):
    batch = sorted_dict[i:i+20]
    for j, (word, freq) in enumerate(batch):
        print(f"{i+j+1}. {word}: {freq}")
    print()


# In[75]:


# most frequent 30 words 1,2,3 star rating youtube
# freq_dict = wordCloud.process_text(consolidated)
# sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
# for i, (word, freq) in enumerate(sorted_dict[:30]):
#     print(f"{i+1}. {word}: {freq}")


# In[ ]:





# In[78]:


#  YOUTUBE 

# youtube 4,5 star rating wordcloud
consolidated=' '.join(word for word in youtubedf2['review'][youtubedf2['sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110, colormap='Paired')
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[79]:


# most frequent 30 words 4,5 star rating youtube
freq_dict = wordCloud.process_text(consolidated)
sorted_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

for i in range(0, min(len(sorted_dict), 40), 20):
    batch = sorted_dict[i:i+20]
    for j, (word, freq) in enumerate(batch):
        print(f"{i+j+1}. {word}: {freq}")
    print()


# In[ ]:





# In[31]:


# tiktok logicstic regression

# shows how relevant a word to the text

# Term Frequency-inverse Document Frequency, statistical method to evaluate importance of a word

#The meaning increases proportionally to the number of times in the text a word appears 
#          but is compensated by the word frequency in the corpus (data-set)


# In[106]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(tiktokdf2['review'] ).toarray()


# In[107]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,tiktokdf2['sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[108]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print('accuracy',accuracy_score(y_test,pred))

# the accuracy for using the review to predict it's sentiment or 1,2,3 star rating or 4,5 star rating


# In[109]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = [False, True])
 
cm_display.plot()
plt.title('Confusion Matrix for Predicted Titktok test_Data')
plt.show()

# 250 predicted test_set of tiktok data


# In[110]:


# youtube logistic regression


# In[111]:


cv_1 = TfidfVectorizer(max_features=2500)
X_1 = cv_1.fit_transform(youtubedf2['review'] ).toarray()


# In[112]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X_1,youtubedf2['sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[113]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print('accuracy',accuracy_score(y_test,pred_1))


# In[114]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = [False, True])
cm_display.plot()
plt.title('Confusion Matrix for Predicted YouTube_Data')
plt.show()

# 250 predicted test_set of youtube data


# In[ ]:





# In[41]:


# Pytrend


# In[80]:


from pytrends.request import TrendReq
pytrends= TrendReq(hl='en-US', tz=360)


# In[96]:


kw_list1 = ["Youtube","TikTok", "Video", "Internet",'Content Creator']
pytrends.build_payload(kw_list1, cat=0, timeframe='today 5-y', geo='')   # 5 years back



# has error run it google colab


# In[97]:


iot = pytrends.interest_over_time()
iot.plot()

# has error in graph run on google colab


# In[ ]:





# In[ ]:




