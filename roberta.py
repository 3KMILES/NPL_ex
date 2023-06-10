# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:30:38 2023


"""

import pandas as pd
from transformers import RobertaForQuestionAnswering, RobertaTokenizer
import torch
#import matplotlib.pyplot as plt
num = 0
# Φορτώστε το αρχείο CSV σε ένα Pandas DataFrame
df = pd.read_csv('path/to/file/steam.csv') # βαλε το σωστο path που βρισκεται το .csv file

# Προεπεξεργασία δεδομένων κειμένου
df['categories'] = df['categories'].str.lower().replace('[^\w\s]','')

# Τοποθετήστε το προεκπαιδευμένο μοντέλο
model = RobertaForQuestionAnswering.from_pretrained('roberta-large')

# Προσαρμογή (Tokenize) δεδομένων κειμένου
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
tokenized_data = tokenizer.batch_encode_plus(df['categories'], pad_to_max_length=True)

# Μετατροπή tokenized δεδομένων σε λειτουργίες εισόδου
input_ids = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']
features = {'input_ids': input_ids, 'attention_mask': attention_masks}

# Χρησιμοποιεί προεκπαιδευμένο μοντέλο RoBERTa για να προβλέψετε απαντήσεις

for i in range(len(df)):
    inputs = {key: torch.tensor(val[i]).unsqueeze(0) for key, val in features.items()}
    #Βάζει τα inputs μέσω του μοντέλου και λάμβανει τα start scores και end scores
    outputs = model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    # Βρίσκει τους δείκτες έναρξης και λήξης της 	απάντησης.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    # Μετατρέψτε τα διακριτικά απαντήσεων σε συμβολοσειρά.
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
    #Εμφανίζει την ερώτηση και την απάντηση
    print('What is the categories of game with index ' + str(num) +  ' ?:' )
    print(f'Answer: {answer}\n')
    num=num + 1
  
   # Plot bar chart
  # df.plot.bar(x='name', y='appid', rot=0)
  # plt.show()