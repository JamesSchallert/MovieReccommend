
# coding: utf-8

# In[1]:


import pandas as pd
from orangecontrib.associate.fpgrowth import *  
import Orange


# In[2]:



movies = pd.read_table('movies.dat')


# In[3]:


movies = movies[(movies['movieID']<50)]


# In[4]:




movieNames = pd.read_table('movieNames.dat')


# In[5]:





viewed = dict()
for row in movies.itertuples():
    if row[1] not in viewed:
        
        viewed[row[1]] = list(movieNames[movieNames['id']==row[2]]['title'])[0]
    else:
        viewed[row[1]] += ', '
        viewed[row[1]] += list(movieNames[movieNames['id']==row[2]]['title'])[0]


# In[6]:



raw_data = list(viewed.values())

f = open('movieData.basket', 'w', encoding='utf-8')
for item in raw_data:
    f.write(item + '\n')
f.close()


# In[7]:


data = Orange.data.Table("movieData.basket")
X, mapping = OneHot.encode(data, include_class=True)


# In[8]:


itemsets = dict(frequent_itemsets(X, .15))


# In[9]:


class_items ={item for item, var, _ in OneHot.decode(mapping, data, mapping)}


# In[11]:


rules = [(P, Q, supp/len(X), conf) for P, Q, supp, conf in association_rules(itemsets,min_confidence = 0.4) if len(Q) == 1 and Q & class_items]


# In[12]:


names = {item: '{}'.format(var.name) for item, var, val in OneHot.decode(mapping, data, mapping)}


# In[29]:





# In[13]:


print('Association Rules:')
for ante, cons, supp, conf in rules:
    print(', '.join(names[i] for i in ante), '-->', names[next(iter(cons))],
    '(supp: {:0.2f}, conf: {:0.2f})'.format(supp, conf))


# In[23]:


userNum = 0
for user in X:
    print('\n\nUser {}'.format(userNum))
    print('Movies Seen:\n{}'.format(', '.join(names[i] for i in user)))
    print('\nPossible Recommendations:')
    recommendations = dict()
    for rule in rules:
        if list(rule[1])[0] not in user:
            if rule[0].issubset(user):
                    ante, cons, supp, conf = rule
                    if names[list(cons)[0]] not in recommendations:
                        recommendations[names[list(cons)[0]]] = 1
                    else:
                        recommendations[names[list(cons)[0]]] += 1
                    print(', '.join(names[i] for i in ante), '-->', names[next(iter(cons))],
                          '(supp: {:0.2f}, conf: {:0.2f})'.format(supp, conf))
    print('\nFrequency of Recommendations:')
    print(recommendations)
        
    
    userNum += 1

