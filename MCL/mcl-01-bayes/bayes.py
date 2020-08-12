#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from pprint import pprint
from math import log, exp


# In[2]:


TRAINSET = "TrainsetTugas1ML.csv"
TESTSET = "TestsetTugas1ML.csv"
TEBAKAN = "TebakanTugas1ML.csv"
OUTPUT_ATTR = "income"


# In[3]:


def read_dict_from_csv(csv_file):
    with open(csv_file) as cf:
        csv_reader = csv.DictReader(cf, skipinitialspace=True)
        data = [dict(row) for row in csv_reader]
    return data


# In[4]:


def write_list_to_csv(csv_file, data):
    with open(csv_file, mode='w') as cf:
        for d in data: cf.write(d + '\n')


# In[5]:


train_data = read_dict_from_csv(TRAINSET)
train_data[:10]


# In[6]:


attributes = {attr: set(d[attr] for d in train_data) for attr in train_data[0] if attr not in {"id", OUTPUT_ATTR}}
attributes


# In[7]:


output_classes = set(d[OUTPUT_ATTR] for d in train_data)
output_classes


# In[8]:


output_frequency = {out: sum(1 for d in train_data if d[OUTPUT_ATTR] == out) for out in output_classes}
output_frequency


# In[9]:


output_probability = {out: output_frequency[out] / len(train_data) for out in output_classes}
output_probability


# In[10]:


class_probability = {
    attr: {
        cls: {
            out: sum(1 for d in train_data if d[attr] == cls and d[OUTPUT_ATTR] == out) / output_frequency[out] 
            for out in output_classes
        } for cls in attributes[attr]
    } for attr in attributes
}
pprint(class_probability)


# In[11]:


test_data = read_dict_from_csv(TESTSET)
test_data[:10]


# In[12]:


for d in test_data:
    out_prob = { out: exp(sum(map(log, (class_probability[attr][d[attr]][out] for attr in attributes)))) * output_probability[out] for out in output_classes }
    d[OUTPUT_ATTR] = max(out_prob, key=out_prob.get)
test_data
output_list = list(map(lambda x: x[OUTPUT_ATTR], test_data))
print(output_list)


# In[13]:


write_list_to_csv(TEBAKAN, output_list)

