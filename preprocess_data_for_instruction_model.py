import pandas as pd
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split

def preprocess_provo_corpus(provo):
    """Reading the raw Provo Corpus dataset and create a dictionary with all useful
    information we might need from it"""
    predict_norms = pd.read_csv(provo, sep='\t')
    paragraphs = predict_norms.groupby('Text_ID')['Text'].max()

    provo_processed = {}
    count = 0
    for text_id in range(1,56): #iterate over all provo paragraphs
        for word_num in predict_norms[predict_norms['Text_ID'] == text_id]['Word_Number'].unique(): #iterating over all words in each text
            word_dist = predict_norms[(predict_norms['Text_ID'] == text_id) & (predict_norms['Word_Number'] == word_num)]
            unique_human_words = word_dist['Response'].unique() #all human answered words for each context
            unique_word_dist = []
            for word in unique_human_words:
                unique_word_count = sum(word_dist[word_dist['Response'] == word]['Response_Count']) #getting all counts of the unique word and summing them
                unique_word_dist.append((word, unique_word_count))

            provo_processed[count] = {}
            provo_processed[count]['context_with_original_word'] = paragraphs[text_id].split(' ')[:int(word_num)]
            provo_processed[count]['context'] = paragraphs[text_id].split(' ')[:(int(word_num)-1)]
            provo_processed[count]['original_positioning'] = {'text_id':text_id, 'word_num':word_num}
            provo_processed[count]['human_next_word_pred'] = unique_word_dist

            count = count + 1

    return provo_processed

input_data = 'Provo_Corpus.tsv'
data = preprocess_provo_corpus(input_data)

all_paragraph_ids = sorted(list(set(sample['original_positioning']['text_id'] for sample in data.values())))

train_val_paragraph_ids, test_paragraph_ids = train_test_split(
    all_paragraph_ids, test_size=0.2, random_state=42)

train_paragraph_ids, val_paragraph_ids = train_test_split(
    train_val_paragraph_ids, test_size=0.1, random_state=42)

instruction_data = {
    'train': [],
    'val': [],
    'test': []
}
# Create lookup
text_id_to_split = {}
for tid in train_paragraph_ids:
    text_id_to_split[tid] = 'train'
for tid in val_paragraph_ids:
    text_id_to_split[tid] = 'val'
for tid in test_paragraph_ids:
    text_id_to_split[tid] = 'test'

# Populate instruction_data with proper split assignment
for inst in data:
    preds = list(data[inst]["human_next_word_pred"])
    text_id = data[inst]['original_positioning']['text_id']

    split_name = text_id_to_split.get(text_id)
    if split_name is None:
        continue  # Skip if text_id wasn't assigned (shouldn't happen)

    for pred in preds:
        for _ in range(pred[1]):
            prompt = " ".join(data[inst]["context"])
            message = {
                'messages': [
                    {'role': 'system', 'content': 'Return five unique plausible next words for the following prompt.'},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': pred[0]}
                ]
            }
            instruction_data[split_name].append(message)

Dataset.from_list(instruction_data['train']).to_json("train_dataset.json", orient="records", lines=True)
Dataset.from_list(instruction_data['val']).to_json("val_dataset.json", orient="records", lines=True)
Dataset.from_list(instruction_data['test']).to_json("test_dataset.json", orient="records", lines=True)

# train_dataset.to_json("train_dataset.json", orient="records", lines=True)
# test_dataset.to_json("test_dataset.json", orient="records", lines=True)