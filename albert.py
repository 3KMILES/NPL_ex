# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:51:22 2023


"""

import pandas as pd
from transformers import AlbertForQuestionAnswering, AlbertTokenizer
import torch


num = 0

# Φορτώστε το αρχείο CSV σε ένα Pandas DataFrame
df = pd.read_csv('path/to/file/steam.csv')  # βαλε το σωστο path που βρισκεται το .csv file

# Προεπεξεργασία δεδομένων κειμένου
df['developer'] = df['developer'].str.lower().replace('[^\w\s]','')

# Τοποθετήστε το προεκπαιδευμένο μοντέλο
model = AlbertForQuestionAnswering.from_pretrained('albert-large-v2')

# Προσαρμογή (Tokenize) δεδομένων κειμένου
tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')

tokenized_data = tokenizer.batch_encode_plus(df['developer'], pad_to_max_length=True)

# Μετατροπή tokenized δεδομένων σε λειτουργίες εισόδου
input_ids = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']
features = {'input_ids': input_ids, 'attention_mask': attention_masks}

# Χρησιμοποιεί προεκπαιδευμένο μοντέλο ALBERT για να προβλέψετε απαντήσεις

for i in range(len(df)):
    inputs = {key: torch.tensor(val[i]).unsqueeze(0) for key, val in features.items()}
    #Βάζει τα inputs μέσω του μοντέλου και λάμβανει τα start scores και end scores
    outputs = model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
   # Βρίσκει τους δείκτες έναρξης και λήξης της απάντησης.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    # Μετατρέψτε τα διακριτικά απαντήσεων σε συμβολοσειρά.
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
    #Εμφανίζει την ερώτηση και την απάντηση
    print('Who is the developer with index ' + str(num) +  ' ?:' )
    print(f'Answer: {answer}\n')
    num=num + 1 # ξεκινάει απο 0 μεχρι το 27075