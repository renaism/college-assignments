#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
from math import log, inf


# ## Baca file corpus dan memisahkan data train dan data test

# In[2]:


df = pd.read_csv('Indonesian_Manually_Tagged_Corpus_ID.tsv', sep='\t', header=None, names=['word','tag'], converters={'word': lambda x: x.lower()}, quoting=csv.QUOTE_NONE)


# In[3]:


split_index1 = int(df[df['word'].str.contains('id=1001')].index[0])
split_index2 = int(df[df['word'].str.contains('id=1021')].index[0])


# In[4]:


df['word'] = df['word'].str.replace(r'<kalimat id=.*>', '<s>')
df['word'] = df['word'].str.replace('</kalimat>', '</s>', regex=False)
df['word'] = df['word'].str.replace(r'(-?\d+.*)', 'NUM')

df.loc[df.word == '<s>', 'tag'] = '<s>'
df.loc[df.word == '</s>', 'tag'] = '</s>'


# In[5]:


df_train, df_test = df[:split_index1].reset_index(drop=True), df[split_index1:split_index2].reset_index(drop=True)


# ## Buat bigram dari urutan tag pada setiap kalimat

# In[6]:


def build_bigram(tokens):
    # Get all the unique tokens
    unique_tokens = set(tokens)

    # Initialize unigram and bigram dictionaries
    unigram = {}
    bigram = {}
    for word_row in unique_tokens:
        unigram[word_row] = 0
        bigram[word_row] = {}
        for word_column in unique_tokens:
            bigram[word_row][word_column] = 0

    # Build the bigram and unigram from word sequences in corpus tokens
    unigram[ tokens[0] ] += 1
    for i in range(1, len(tokens)):
        unigram[ tokens[i] ] += 1
        bigram[ tokens[i-1] ][ tokens[i] ] += 1

    # Laplace (add-one) smoothing
#     for word_row in bigram:
#         # Exclude sentence-end flag because it's always followed by sentence-start flag
#         if word_row == "</s>":
#             continue
#         # row <s> shouldn't has column <s> and </s> so it's decremented by 2
#         unigram[word_row] += len(unigram) if word_row != "<s>" else len(unigram) - 2
#         for word_column in bigram[word_row]:
#             bigram[word_row][word_column] += 1

    # Normalize the bigram with unigram
    for word_row in bigram:
        for word_column in bigram[word_row]:
            bigram[word_row][word_column] /= unigram[word_row]
    
    # Special case handling for flags
    bigram["<s>"]["</s>"] = 0.0
    bigram["<s>"]["<s>"] = 0.0
    bigram["</s>"]["<s>"] = 1.0
    
    return bigram


# In[7]:


tokens = list(df_train['tag'])
tokens


# In[8]:


tag_bigram = pd.DataFrame.from_dict(build_bigram(tokens), orient='index')
tag_bigram


# ## Tabel emisi tag-kata

# In[9]:


tabel = df_train.groupby(['word', 'tag']).size().unstack(fill_value=0)
tabel


# In[10]:


tabel1 = tabel.astype('float')
for i, word in tabel1.iterrows():
    for tag in word.index:
        if (word[tag] > 0):
            word[tag] = float(word[tag]) / tabel1.loc[:, tag].sum()
tabel1


# ## Baseline model

# In[11]:


df_test


# In[12]:


bener = 0
salah = 0
not_exist = 0
all = 0
for i, row in df_test.iterrows():
#     print(row['word'])
    if (row['word'] in tabel.index):
        if (row['tag'] == tabel.loc[row['word']].idxmax()):
            bener += 1
        else :
            salah += 1
    else :
        not_exist += 1
    all += 1
print('bener :',bener,', salah :',salah,', ga ada :', not_exist,', semua :', all)


# In[13]:


baseline_acc_loss = bener / (all) * 100
print(baseline_acc_loss, '%')
baseline_acc = bener / (all - not_exist) * 100
print(baseline_acc, '%')


# In[14]:


for i, row in df_test.iterrows():
    print(row['word'] in tabel.index, row['word'], row['tag'])


# <h2>HMM Model</h2>

# In[15]:


def viterbi(words, tags, trans_p, emit_p):
    V = [{}]
    # Buat kata/tag pertama di awal kalimat
    for tag in tags:
        word_p = emit_p.loc[words[0], tag]
        # Simpan hanya yang probabilitasnya lebih dari 0 untuk kata pertama ber-tag 'tag'
        if word_p > 0:
            tr = trans_p.loc['<s>', tag]
            V[0][tag] = {"prob": (log(tr) if tr > 0 else -inf) + log(word_p), 'prev': None}
    # Kata/tag selanjutnya
    for i in range(1, len(words)):
        V.append({})
        word_known = words[i] in tabel1.index
        valid_tags = list(filter(lambda x: emit_p.loc[words[i], x] > 0, tags)) if word_known else ['X']
        for tag in valid_tags:
            max_tr_prob = -inf
            prev_tag_selected = None
            for prev_tag in V[i-1]:
                tr = trans_p.loc[prev_tag, tag]
                tr_prob = V[i-1][prev_tag]['prob'] + (log(tr) if tr > 0 else -inf)
                if (prev_tag_selected == None or tr_prob > max_tr_prob):
                    max_tr_prob = tr_prob
                    prev_tag_selected = prev_tag
            max_prob = max_tr_prob + log(emit_p.loc[words[i], tag]) if word_known else max_tr_prob
            #if (max_prob == -inf):
             #   prev_tag_selected = tabel.loc[words[i-1]].idxmax() if words[i-1] in tabel1.index else 'X'
            V[i][tag] = {'prob': max_prob, 'prev': prev_tag_selected}
    best_tags = []
    
    # Cari end-point dengan probabilitas maksimal
    max_end_prob = max(x['prob'] for x in V[-1].values())
    #print('end', max_end_prob)
    prev_tag = None
    
    for tag, data in V[-1].items():
        if data['prob'] == max_end_prob:
            best_tags.append(tag)
            prev_tag = tag
            #print(tag)
            break
            
    # Backtrack dari end-point ke awal
    for i in range(len(V)-2, -1, -1):
        #print(V[i+1])
        best_tags.insert(0, V[i+1][prev_tag]['prev'])
        prev_tag = V[i+1][prev_tag]['prev']
    return best_tags


# In[16]:


words = []
original_tag = []
hmm_tag = []
knowns = []
for i, row in df_test.iterrows():
    words.append(row['word'])
    knowns.append(row['word'] in tabel.index)
    original_tag.append(row['tag'])
    if (row['tag'] == '</s>'):
        hmm_tag.extend(viterbi(words, tokens, tag_bigram, tabel1))
        words = []
hmm_result = pd.DataFrame(list(zip(list(df_test['word']), knowns, original_tag, hmm_tag)), columns=['word', 'known_word', 'original_tag', 'hmm_tag'])
hmm_result


# In[18]:


total = len(hmm_result.index)
correct = 0
incorrect = 0
known = 0
for i, tag in enumerate(hmm_tag):
    if knowns[i]:
        known += 1
        if hmm_tag[i] == original_tag[i]:
            correct += 1
        else:
            incorrect += 1
hmm_acc_loss = correct / total * 100
hmm_acc = correct / known * 100

print('bener :',correct,', salah :',incorrect,', ga ada :', total-known,', semua :', total)
print(hmm_acc_loss, '%')
print(hmm_acc, '%')


# ## Perbandingan Akurasi

# ### Baseline

# In[20]:


print('Include not found: ', baseline_acc_loss, '%')
print('Found only: ', baseline_acc, '%')


# ### HMM

# In[21]:


print('Include not found: ', hmm_acc_loss, '%')
print('Found only: ', hmm_acc, '%')

