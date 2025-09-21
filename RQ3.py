from datasets import load_dataset
from huggingface_hub import login
import json
import numpy as np
import random
from collections import Counter
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import re
import csv
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from huggingface_hub import login
# login("VALID LOGIN CREDENTIALS HERE")

"""
This all relates loading and editting the stanfordnlp web questions dataset
"""
# ds = load_dataset("stanfordnlp/web_questions")
# ds['test'].to_json("RQ3data.json")
# # ds = ds.select(range(100))

# output_path = "filtered_output.json"

# with open("RQ3data.json", "r", encoding="utf-8") as f, open(output_path, 'w', encoding='utf-8') as outfile:
#     for line in f:
#         item = json.loads(line)
#         question = item.get("question", "")
#         answers = item.get("answers", [])
#         for ans in answers:
#             if isinstance(ans, str) and len(ans.split()) == 1:
#                 result = {
#                     "question": question,
#                     "answer": ans
#                 }
#                 json.dump(result, outfile)
#                 outfile.write("\n")


# data = load_dataset('json', data_files='RQ3_data.json', split='train')
# instruction_data = []
# for inst in data:
#     pred = inst["next_word"]
#     prompt = inst["content"]
#     message = {
#         'messages': [
#             {'role': 'system', 'content': 'Only return one plausible next word for the following the sentence.'},
#             {'role': 'user', 'content': prompt},
#             {'role': 'assistant', 'content': pred}
#         ]
#     }
#     instruction_data.append(message)

# with open('RQ3I_Data.json', 'w') as f:
#     json.dump(instruction_data, f, indent=4)


# group all samples with same context together
def group_by_context(test_data):
    grouped = defaultdict(list)
    for sample in test_data:
        context = sample['messages'][1]['content']
        next_word = sample['messages'][2]['content'].strip().lower()
        grouped[context].append(next_word)
    return grouped

def compute_distribution(elements):
    c = Counter(elements)
    mle = {word: count / sum(c.values()) for word, count in c.items()}

    return mle

def entropy(distribution):
    return -sum(p * np.log2(p) for p in distribution.values())

# get average distributions across seeds
def average_distributions(distributions_list):
    averaged = {}
    for key in distributions_list[0].keys():
        temp = defaultdict(float)
        for dist in distributions_list:
            for k, v in dist[key].items():
                temp[k] += v
        averaged[key] = {k: v / len(distributions_list) for k, v in temp.items()}
    return averaged

def get_common_support(support1, support2):
    """Receives supports from two distributions and return all elements appearing in at least one of them"""
    return set(support1).union(set(support2))

def change_support(old_support, old_probs, new_support):
    """Create new support by adding elements to a support that did not exist before
     (hence, their probability value is 0)"""
    new_probs = []
    for item in new_support:
        if item in old_support:
            ind = old_support.index(item)
            new_probs.append(old_probs[ind])
        else:
            new_probs.append(0)
    return list(new_support), new_probs

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

# instruction-tuned models repeat the instruction in their output before they return their next word preds
def extract_word_after_context(reply, context):
    reply_clean = clean_text(reply)
    context_clean = clean_text(context)

    # Find where context ends in the reply
    reply_words = reply_clean.split()
    context_words = context_clean.split()

    for i in range(len(reply_words) - len(context_words)):
        if reply_words[i:i + len(context_words)] == context_words:
            after_index = i + len(context_words)
            if after_index < len(reply_words):
                return reply_words[after_index]
            else:
                return ''  
    return ''


def get_first_word_sample(model, tokenizer, contexts, add_tokens=5, device=None):
    system_prompt = "Only return one plausible next word for the following the sentence."

    # Build prompt messages
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        for context in contexts
    ]

    # Convert messages to prompts
    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    # Tokenize in batch
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate one token sequence per input
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        do_sample=True,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=add_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Postprocess to extract only the next word after context
    results = []
    for decoded, context in zip(decoded_outputs, contexts):
        reply = decoded.split("[/INST]")[-1].strip()
        word = extract_word_after_context(reply, context)
        if not word:
            results.append("Failed to generate word")
        else:
            results.append(word.lower())

    return results

def get_model_samples(model, tokenizer, context, device, n_samples = 5, add_tokens = 5, top_k = 0, pad_token = "<|endoftext|>"):
    """Given a context we return the words that were generated during ancestral sampling for n_samples"""
    tokenized_inputs = tokenizer(context, return_tensors="pt") # Get the full output dict
    inputs_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device) # Get attention_mask and send to device

    #This samples unbiasedly from the next-token distribution of the model for add_tokens tokens, n_samples times
    # outputs = model.generate(inputs_ids, attention_mask = attention_mask, pad_token_id = tokenizer.eos_token_id, do_sample=True, num_beams = 1, num_return_sequences= n_samples,
                            #  max_new_tokens = add_tokens, pad_token = tokenizer.pad_token, top_k= top_k)
    outputs = model.generate(inputs_ids, attention_mask = attention_mask, pad_token_id = tokenizer.eos_token_id, do_sample=True, num_beams = 1, num_return_sequences= n_samples,
                             max_new_tokens = add_tokens)

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #Remove context to keep only generated text
    outputs = [x.replace(context + ' ', '').replace(context, '').replace('\n', '') for x in outputs]
    #Remove punctuation
    outputs = [re.sub(r'[^\w\s]', '', x) for x in outputs] #removing punctuation
    list_of_words = [x.split(' ') for x in outputs]

    sampled_words = []
    for generation in list_of_words:
        if set(generation) == {''}:
            sampled_words.append('Failed to generate word')
        else:
            sampled_words.append(next(x.lower() for x in generation if x))

    return sampled_words


def evaluate_output_distributions(model, tokenizer, test_data, device, num_model_samples=40):
    dist_per_prompt  = {}                  
    model_outputs = {}              
    target_frequency = {} 

    for sample in test_data:
        context = sample['messages'][1]['content']
        target = sample['messages'][2]['content'].strip().lower()       
        model_preds = get_model_samples(model, tokenizer, context, device, n_samples=num_model_samples)
        model_preds = [pred.lower() for pred in model_preds if pred != 'Failed to generate word']
        
        if context not in model_outputs:
            model_outputs[context] = []
        model_outputs[context].extend(model_preds)

        dist_per_prompt[context] = compute_distribution(model_preds)
        
        # Calculate target frequency
        target_count = model_preds.count(target.lower())
        target_frequency[context] = {
            'count': target_count,
            'frequency': target_count / len(model_preds) if model_preds else 0,
            'target_word': target.lower()
        }

    return dist_per_prompt, model_outputs, target_frequency

def evaluate_IT_output_distributions(model, tokenizer, test_data, device, num_model_samples=40):
    dist_per_prompt  = {}                  
    model_outputs = {}              
    target_frequency = {} 

    for sample in test_data:
        context = sample['messages'][1]['content']
        target = sample['messages'][2]['content'].strip().lower()       
        contexts_batch = [context] * num_model_samples
        model_preds = get_first_word_sample(model, tokenizer, contexts_batch, device=device)
        model_preds = [pred.lower() for pred in model_preds if pred != 'Failed to generate word']
        
        if context not in model_outputs:
            model_outputs[context] = []
        model_outputs[context].extend(model_preds)

        dist_per_prompt[context] = compute_distribution(model_preds)
        
        # Calculate target frequency
        target_count = model_preds.count(target.lower())
        target_frequency[context] = {
            'count': target_count,
            'frequency': target_count / len(model_preds) if model_preds else 0,
            'target_word': target.lower()
        }

    return dist_per_prompt, model_outputs, target_frequency


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = load_dataset('json', data_files='RQ3I_Data.json', split='train')
num_model_samples = 40

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

seeds = [42, 123, 456]

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
base_model.to(device)
base_model.eval()

# Save base model/tokenizer if needed
if not os.path.exists("./mistral7b"):
    base_model.save_pretrained("./mistral7b")
    tokenizer.save_pretrained("./mistral7b")


# --- Evaluate Non-fine-tuned Model ---
target_counts_non_fine_tuned = {}
non_fine_tuned_distr = []
full_tf_non_fine_tuned = {}
mean_tf_non_fine_tuned = {}

it_target_counts_non_fine_tuned = {}
it_non_fine_tuned_distr = []
it_full_tf_non_fine_tuned = {}
it_mean_tf_non_fine_tuned = {}

for seed in seeds:
    print(f"----- Starting Non-fine-tuned Seed {seed} -----")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    non_ft_model_path =  f'gpt2_seed_{seed}'
    # Load fine-tuned model from saved path
    model_non_ft = GPT2LMHeadModel.from_pretrained(non_ft_model_path)
    model_non_ft.to(device)
    # GPT-2
    dist_per_context, model_outputs, target_freq = evaluate_output_distributions(model_non_ft, gpt_tokenizer, test_data, device, num_model_samples=num_model_samples)
    non_fine_tuned_distr.append(dist_per_context)
    target_counts_non_fine_tuned[seed] = [target_freq['count'] for target_freq in target_freq.values()]
    full_tf_non_fine_tuned[seed] = [target_freq['frequency'] for target_freq in target_freq.values()]
    mean_tf_non_fine_tuned[seed] = np.mean(full_tf_non_fine_tuned[seed])

    # Instruction-tuned
    dist_per_context, model_outputs, target_freq = evaluate_IT_output_distributions(base_model, tokenizer, test_data, device, num_model_samples=num_model_samples)
    it_non_fine_tuned_distr.append(dist_per_context)
    it_target_counts_non_fine_tuned[seed] = [target_freq['count'] for target_freq in target_freq.values()]
    it_full_tf_non_fine_tuned[seed] = [target_freq['frequency'] for target_freq in target_freq.values()]
    it_mean_tf_non_fine_tuned[seed] = np.mean(it_full_tf_non_fine_tuned[seed])

    # print(f"Mean Target Frequency (Non-fine-tuned) for seed {seed}: {mean_tf_non_fine_tuned[seed]}")

    # Save target frequency results
    with open(f'q3_target_frequency_non_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, freq_data in target_freq.items():
            f.write(f"Context: {context}\n")
            f.write(f"Target word: {freq_data['target_word']}\n")
            f.write(f"Count: {freq_data['count']}/{num_model_samples}\n")
            f.write(f"Frequency: {freq_data['frequency']:.4f}\n")
            f.write("-" * 50 + "\n")

    with open(f'q3_model_outputs_non_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in model_outputs.items():
            f.write(f"{context} -> {preds}\n")


# --- Evaluate Fine-tuned Models ---
print("\n----- Evaluating Fine-tuned Models -----")
fine_tuned_dist = []
mean_tf_fine_tuned = {}
full_tf_fine_tuned = {}
target_counts_fine_tuned = {}

it_fine_tuned_dist = []
it_mean_tf_fine_tuned = {}
it_full_tf_fine_tuned = {}
it_target_counts_fine_tuned = {}

for seed in seeds:
    print(f"\n----- Fine-tuned Seed {seed} -----")

    ft_model_path =  f'fine_tuned_gpt2_seed_{seed}'
    # Load fine-tuned model from saved path
    model_ft = GPT2LMHeadModel.from_pretrained(ft_model_path)
    model_ft.to(device)

    model_path = f"./fine_tuned_mistral7b_seed_{seed}"

    # Load tokenizer & base model fresh each time
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = PeftModel.from_pretrained(model, model_path, config=lora_config)
    model.to(device)
    model.eval()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPT-2
    dist_per_context, model_outputs, target_freq = evaluate_output_distributions(model_ft, gpt_tokenizer, test_data, device, num_model_samples=num_model_samples)
    fine_tuned_dist.append(dist_per_context)
    target_counts_fine_tuned[seed] = [target_freq['count'] for target_freq in target_freq.values()]
    full_tf_fine_tuned[seed] = [target_freq['frequency'] for target_freq in target_freq.values()]
    mean_tf_fine_tuned[seed] = np.mean(full_tf_fine_tuned[seed])

    # Instruction-tuned
    dist_per_context, model_outputs, target_freq = evaluate_IT_output_distributions(model, tokenizer, test_data, device, num_model_samples=num_model_samples)
    it_fine_tuned_dist.append(dist_per_context)
    it_target_counts_fine_tuned[seed] = [target_freq['count'] for target_freq in target_freq.values()]
    it_full_tf_fine_tuned[seed] = [target_freq['frequency'] for target_freq in target_freq.values()]
    it_mean_tf_fine_tuned[seed] = np.mean(it_full_tf_fine_tuned[seed])

    print(f"Mean Target Frequency (Fine-tuned) for seed {seed}: {mean_tf_fine_tuned[seed]}")

    # Save target frequency results
    with open(f'target_frequency_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, freq_data in target_freq.items():
            f.write(f"Context: {context}\n")
            f.write(f"Target word: {freq_data['target_word']}\n")
            f.write(f"Count: {freq_data['count']}/{num_model_samples}\n")
            f.write(f"Frequency: {freq_data['frequency']:.4f}\n")
            f.write("-" * 50 + "\n")

    with open(f'q3_model_outputs_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in model_outputs.items():
            f.write(f"{context} -> {preds}\n")


print("\n=== Evaluation Complete. ===")

csv_data = []
for seed in seeds:
    row = {
        'Seed': seed,
        'Mean TF Non-Fine-Tuned': float(mean_tf_non_fine_tuned[seed]),
        'Mean TF Fine-Tuned': float(mean_tf_fine_tuned[seed]),
        'Full TF Non-Fine-Tuned': str(full_tf_non_fine_tuned[seed]),
        'Full TF Fine-Tuned': str(full_tf_fine_tuned[seed]),
    }
    csv_data.append(row)

with open('RQ3_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Seed', 'Mean TF Non-Fine-Tuned', 'Mean TF Fine-Tuned', 
        'Full TF Non-Fine-Tuned', 'Full TF Fine-Tuned'])
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)

print("\nResults saved to RQ3_results.csv")

it_csv_data = []
for seed in seeds:
    row = {
        'Seed': seed,
        'Mean TF Non-Fine-Tuned': float(it_mean_tf_non_fine_tuned[seed]),
        'Mean TF Fine-Tuned': float(it_mean_tf_fine_tuned[seed]),
        'Full TF Non-Fine-Tuned': str(it_full_tf_non_fine_tuned[seed]),
        'Full TF Fine-Tuned': str(it_full_tf_fine_tuned[seed]),
    }
    it_csv_data.append(row)

with open('IT_RQ3_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Seed', 'Mean TF Non-Fine-Tuned', 'Mean TF Fine-Tuned', 
        'Full TF Non-Fine-Tuned', 'Full TF Fine-Tuned'])
    writer.writeheader()
    for row in it_csv_data:
        writer.writerow(row)

print("\nResults saved to RQ3_results.csv")

print("\n----- Comparison of Fine-tuned vs Non-fine-tuned Models -----")
seed_mean_non_ft_mean = []
seed_mean_ft_mean = []
it_seed_mean_non_ft_mean = []
it_seed_mean_ft_mean = []
for seed in seeds:
    non_ft_mean = csv_data[seeds.index(seed)]['Mean TF Non-Fine-Tuned']
    ft_mean = csv_data[seeds.index(seed)]['Mean TF Fine-Tuned']
    seed_mean_non_ft_mean.append(non_ft_mean)
    seed_mean_ft_mean.append(ft_mean)
    
    it_non_ft_mean = it_csv_data[seeds.index(seed)]['Mean TF Non-Fine-Tuned']
    it_ft_mean = it_csv_data[seeds.index(seed)]['Mean TF Fine-Tuned']
    it_seed_mean_non_ft_mean.append(it_non_ft_mean)
    it_seed_mean_ft_mean.append(it_ft_mean)
    print(f"Seed {seed}:")
    print(f"  Non-fine-tuned TF: {non_ft_mean}")
    print(f"  Fine-tuned : {ft_mean}")
    print(f"  Non-fine-tuned instruction-tuned TF: {it_non_ft_mean}")
    print(f"  Fine-tuned instruction-tuned TF: {it_ft_mean}")



# VALUES FOR TABLE
print(f"Seed Mean Non-Fine-Tuned: {np.mean(seed_mean_non_ft_mean)}")
print(f"Seed Mean Fine-Tuned: {np.mean(seed_mean_ft_mean)}")
print(f"Seed Std Non-Fine-Tuned: {np.std(seed_mean_non_ft_mean):.4f}")
print(f"Seed Std Fine-Tuned: {np.std(seed_mean_ft_mean):.4f}")

print(f"Seed Mean IT Non-Fine-Tuned: {np.mean(it_seed_mean_non_ft_mean)}")
print(f"Seed Mean IT Fine-Tuned: {np.mean(it_seed_mean_ft_mean)}")
print(f"Seed Std IT Non-Fine-Tuned: {np.std(it_seed_mean_non_ft_mean):.4f}")
print(f"Seed Std IT Fine-Tuned: {np.std(it_seed_mean_ft_mean):.4f}")



mean_non_ft_freqs = np.mean([full_tf_non_fine_tuned[seed] for seed in seeds], axis=0)
mean_ft_freqs = np.mean([full_tf_fine_tuned[seed] for seed in seeds], axis=0)

it_mean_non_ft_freqs = np.mean([it_full_tf_non_fine_tuned[seed] for seed in seeds], axis=0)
it_mean_ft_freqs = np.mean([it_full_tf_fine_tuned[seed] for seed in seeds], axis=0)

# HISTOGRAM OF HIT RATES
def target_hit_rate_histogram(tf_scores_non_fine_tuned, tf_scores_fine_tuned, name):
    """
    Plot the TVD scores for both models in a histogram.
    """
    plt.clf()  # Clear the current figure
    # Filter out NaN values
    tvd_scores_non_fine_tuned_cleaned = [s for s in tf_scores_non_fine_tuned if pd.notna(s)]
    tvd_scores_fine_tuned_cleaned = [s for s in tf_scores_fine_tuned if pd.notna(s)]

    # Define colors to use for facecolor and edgecolor
    # These are the typical first three default colors in matplotlib's cycle
    color_nft = 'C0'
    color_ft = 'C1'

    # Plot TVD scores in a histogram
    plt.hist(tvd_scores_non_fine_tuned_cleaned, bins=20, alpha=0.5, label='Non-Fine-Tuned Model',
             color=color_nft, edgecolor=color_nft, hatch='\\')
    plt.hist(tvd_scores_fine_tuned_cleaned, bins=20, alpha=0.5, label='Fine-Tuned Model',
             color=color_ft, edgecolor=color_ft, hatch='o')

    plt.title("Histogram of Target Hit Rates (Non-Fine-Tuned vs. Fine-Tuned Model)")
    plt.xlabel("Hit Rates")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.show()

target_hit_rate_histogram(mean_non_ft_freqs, mean_ft_freqs, "hit_rate_comparison_gpt")

target_hit_rate_histogram(it_mean_non_ft_freqs, it_mean_ft_freqs, "hit_rate_comparison_it")


non_finetuned_avg_output_distributions = average_distributions(non_fine_tuned_distr)
non_finetuned_entropy = {prompt: entropy(dist) for prompt, dist in non_finetuned_avg_output_distributions.items()}

finetuned_avg_output_distributions = average_distributions(fine_tuned_dist)
finetuned_entropy = {prompt: entropy(dist) for prompt, dist in finetuned_avg_output_distributions.items()}

common_prompts = set(non_finetuned_entropy.keys()) & set(finetuned_entropy.keys())
x = [non_finetuned_entropy[p] for p in common_prompts]
y = [finetuned_entropy[p] for p in common_prompts]

# Calculate Mistral-7B entropy
it_non_finetuned_avg_output_distributions = average_distributions(it_non_fine_tuned_distr)
it_non_finetuned_entropy = {prompt: entropy(dist) for prompt, dist in it_non_finetuned_avg_output_distributions.items()}

it_finetuned_avg_output_distributions = average_distributions(it_fine_tuned_dist)
it_finetuned_entropy = {prompt: entropy(dist) for prompt, dist in it_finetuned_avg_output_distributions.items()}

it_common_prompts = set(it_non_finetuned_entropy.keys()) & set(it_finetuned_entropy.keys())
it_x = [it_non_finetuned_entropy[p] for p in it_common_prompts]
it_y = [it_finetuned_entropy[p] for p in it_common_prompts]

# Create single figure with both models
plt.figure(figsize=(10, 8))

# Plot GPT-2 data
plt.scatter(x, y, marker='x', alpha=0.6, s=50, label='GPT-2', color='blue')
plt.plot([min(x+y), max(x+y)], [min(x+y), max(x+y)], '--', linewidth=2, color='blue', alpha=0.3)

# Plot Mistral-7B data
plt.scatter(it_x, it_y, marker='o', alpha=0.6, s=50, label='Mistral-7B', color='red')
plt.plot([min(it_x+it_y), max(it_x+it_y)], [min(it_x+it_y), max(it_x+it_y)], '--', linewidth=2, color='red', alpha=0.3)

plt.xlabel("Entropy non-fine-tuned")
plt.ylabel("Entropy fine-tuned")
plt.title("Per-Prompt Output Diversity Comparison")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.savefig("entropy_comparison.png")
plt.show()