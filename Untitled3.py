#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


book=pd.read_csv(r"C:\Users\kittu\OneDrive\Documents\project_dataset\Books.csv")


# In[3]:


get_ipython().system('pip install matplotlib')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


## Replacing Invalid years with max year
from collections import Counter
count = Counter(book['Year-Of-Publication'])
[k for k, v in count.items() if v == max(count.values())]
# In[4]:


pd.set_option('display.max_colwidth', -1)


# In[5]:


## Replacing Invalid years with max year
from collections import Counter
count = Counter(book['Year-Of-Publication'])
[k for k, v in count.items() if v == max(count.values())]


# In[6]:


book.loc[book.ISBN == '0789466953','Year-Of-Publication'] = 2000
book.loc[book.ISBN == '0789466953','Book-Author'] = "James Buckley"
book.loc[book.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
book.loc[book.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Book...	"


# In[7]:


book.loc[book.ISBN == '078946697X','Year-Of-Publication'] = 2000
book.loc[book.ISBN == '078946697X','Book-Author'] = "JMichael Teitelbaum"
book.loc[book.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
book.loc[book.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Beg..."


# In[8]:


book.loc[book.ISBN == '2070426769','Year-Of-Publication'] = 2003
book.loc[book.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
book.loc[book.ISBN == '2070426769','Publisher'] = "Gallimard"
book.loc[book.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de Les Bergers"


# In[9]:


## Converting year of publication in Numbers
book['Year-Of-Publication'] = book['Year-Of-Publication'].astype(int)


# In[10]:


book.loc[book['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2000
book.loc[book['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2000


# In[11]:


User=pd.read_csv(r"C:\Users\kittu\OneDrive\Documents\Users.csv")


# In[12]:


User["country"]=User.Location.str.split(",",expand=True)[2]


# In[13]:


User=User.drop(["Location"],axis=1)


# In[14]:


User.loc[User["Age"]>100,"Age"]=np.NaN
User.loc[User["Age"]<6,"Age"]=np.NaN


# In[15]:


User["Age"].fillna(30,inplace = True)


# In[16]:


User["Age"]=User["Age"].astype(int)


# In[17]:


User.dropna()#there are only two null value in country therefore droping it


# In[18]:


rating=pd.read_csv(r"C:\Users\kittu\OneDrive\Documents\Ratings.csv")


# In[19]:


import re
## removing extra characters from ISBN (from ratings dataset) existing in books dataset
bookISBN = book['ISBN'].tolist() 
reg = "[^A-Za-z0-9]" 
for index, row_Value in rating.iterrows():
    z = re.search(reg, row_Value['ISBN'])    
    if z:
        f = re.sub(reg,"",row_Value['ISBN'])
        if f in bookISBN:
            rating.at[index , 'ISBN'] = f


# In[20]:


# merging all data sets
dataset = pd.merge(book, rating, on='ISBN', how='inner')
dataset = pd.merge(dataset, User, on='User-ID', how='inner')
dataset.info()


# In[21]:


p = {}
for year in book['Year-Of-Publication']:
    if str(year) not in p:
        p[str(year)] = 0
    p[str(year)] +=1

p = {k:v for k, v in sorted(p.items())}


# In[22]:


#popular based 
def popularity_based(dataframe, n):
    if n >= 1 and n <= len(dataframe):
        data = pd.DataFrame(dataframe.groupby('ISBN')['Book-Rating'].count()).sort_values('Book-Rating', ascending=False).head(n)
        data1=pd.DataFrame(dataframe.groupby('ISBN')['Book-Rating'].unique())
        data2=pd.merge(data,data1, on='ISBN')
        result = pd.merge(data2, book, on='ISBN', left_index = False)
        return result
    return "Invalid number of books entered!!"


# In[23]:


number = int(input("Enter number of books to recommend: "))


# In[24]:


dataset1=dataset[dataset["Book-Rating"]>8]


# In[25]:


print("Top", number, "Popular books are: ")
popularity_based(dataset1,5)


# In[26]:


popularity_threshold = 50
get_ipython().system('pip install scipy')
get_ipython().system('pip install sklearn')
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
data = (dataset.groupby(by = ['Book-Title'])['Book-Rating'].count().reset_index().
        rename(columns = {'Book-Rating': 'Total-Rating'}))

result = pd.merge(data, dataset, on='Book-Title', left_index = False)
result = result[result['Total-Rating'] >= popularity_threshold]
result = result.reset_index(drop = True)

matrix = result.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
up_matrix = csr_matrix(matrix)


# In[30]:


bookName = input("Enter a book name: ")
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(up_matrix)

distances, indices = model.kneighbors(matrix.loc[bookName].values.reshape(1, -1), n_neighbors =10)
print("\nRecommended books:\n")
for i in range(0, len(distances.flatten())):
    if i > 0:
        print(matrix.index[indices.flatten()[i]]) 


# In[31]:


import pickle
pickle.dump(matrix,open("books.pk1","wb"))
pickle.dump(model,open("model.pk1","wb"))
pickle.dump(result,open("result.pk1","wb"))


# In[32]:


import streamlit as st
import pickle
import pandas as pd
import requests

st.title('books Recommender System')



matrix= pickle.load(open("books.pk1","rb"))
model= pickle.load(open("model.pk1",'rb'))
result= pickle.load(open("result.pk1",'rb'))
def recommend(books):



  distances, indices = model.kneighbors(matrix.loc[books].values.reshape(1, -1), n_neighbors =10)
  print("\nRecommended books:\n")
  for i in range(0, len(distances.flatten())):
    if i > 0:
        print(matrix.index[indices.flatten()[i]]) 



books_list = result['Book-Title'].values
selected_book= st.selectbox('Select a books from drop down',books_list)

st.write('You selected:', selected_book)


if st.button('Show Recommend book'):
    recommended_book_names = recommend(selected_book)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_book_names[0])
        

    with col2:
        st.text(recommended_book_names[1])
      

    with col3:
        st.text(recommended_book_names[2])
      

    with col4:
        st.text(recommended_book_names[3])
       

    with col5:
        st.text(recommended_book_names[4])


# In[ ]:




