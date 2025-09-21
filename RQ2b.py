import numpy as np
import random
from collections import Counter
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import re
import json 
import csv
from collections import defaultdict, Counter

from huggingface_hub import login
# login("VALID LOGIN CREDENTIALS HERE")

def get_estimator(elements):
    """Get the MLE estimate given all words (probability of a word equals its relative frequency)"""
    c = Counter(elements)
    support = list(c.keys())
    counts = list(c.values())
    probs = [count / sum(counts) for count in counts]

    return (support, probs)


def group_by_context(test_data):
    grouped = defaultdict(list)
    for sample in test_data:
        context = sample['messages'][1]['content']
        next_word = sample['messages'][2]['content'].strip().lower()
        grouped[context].append(next_word)
    return grouped

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

def get_tvd(probs1, probs2):
    """Receives the probabilities of 2 distributions to compare and returns their TVD (Total Variation Distance)"""
    tvd = np.sum(np.abs(np.array(probs1) - np.array(probs2)))/2
    return tvd

def get_oracle_elements(words, seed = 0):
    """We receive a list of words and we create two disjoint subsets
    from it by sampling without replacement from them.
    We return the two disjoint sets of words"""
    random.seed(seed)

    #if the length of the list is odd, we remove one element at random to make the list even,
    #to ensure the two disjoint subsets are of equal length
    if (len(words) % 2 == 1):
        remove_word = random.sample(words, 1)
        words.remove(remove_word[0])

    #We sample the words that will belong in the first subset and create the second subset by removing
    #from the full word list the ones sampled in the first subset
    subset1 = random.sample(words, len(words)//2)
    subset2 = words.copy()
    for item in subset1:
        subset2.remove(item)

    return subset1, subset2

def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

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


def evaluate_model_tvd(model, tokenizer, test_data, device, num_model_samples=40):
    tvd_scores = []                  
    model_outputs = {}              
    tvd_per_context = {}            

    grouped_data = group_by_context(test_data)

    for context, human_words in grouped_data.items():
        contexts_batch = [context] * num_model_samples
        model_preds = get_first_word_sample(model, tokenizer, contexts_batch, device=device)
        model_preds = [pred.lower() for pred in model_preds if pred != 'Failed to generate word']
        
        if context not in model_outputs:
            model_outputs[context] = []
        model_outputs[context].extend(model_preds)

        support_human, probs_human = get_estimator(human_words)
        support_model, probs_model = get_estimator(model_preds)

        common_support = get_common_support(support_human, support_model)
        support_human, probs_human = change_support(support_human, probs_human, common_support)
        support_model, probs_model = change_support(support_model, probs_model, common_support)

        tvd = get_tvd(probs_human, probs_model)
        tvd_scores.append(tvd)
        tvd_per_context[context] = tvd     # Save per-context TVD

    return tvd_scores, model_outputs, tvd_per_context

seeds = [42, 123, 456]
test_data = load_dataset('json', data_files='test_dataset.json', split='train')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
tvd_scores_non_fine_tuned = {}
model_outputs_non_fine_tuned = {}
tvd_per_context_non_fine_tuned = {}
mean_tvd_non_fine_tuned = []

# Dictionary to store overlapping outputs
overlapping_outputs = {}

for seed in seeds:
    print(f"----- Starting Non-fine-tuned Seed {seed} -----")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tvd_scores, outputs, tvd_context = evaluate_model_tvd(base_model, tokenizer, test_data, device, num_model_samples=40)

    tvd_scores_non_fine_tuned[seed] = tvd_scores
    model_outputs_non_fine_tuned[seed] = outputs
    tvd_per_context_non_fine_tuned[seed] = tvd_context
    mean_val = np.mean(tvd_scores)
    mean_tvd_non_fine_tuned.append(mean_val)
    print(f"Mean TVD (Non-fine-tuned) for seed {seed}: {mean_val}")

    with open(f'model_outputs_non_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in outputs.items():
            f.write(f"{context} -> {preds}\n")

    with open(f'tvd_per_context_non_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, tvd in tvd_context.items():
            f.write(f"{context}\t{tvd:.6f}\n")


# --- Evaluate Fine-tuned Models ---
print("\n----- Evaluating Fine-tuned Models -----")
tvd_scores_fine_tuned = {}
model_outputs_fine_tuned = {}
tvd_per_context_fine_tuned = {}
mean_tvd_fine_tuned = []

for seed in seeds:
    print(f"\n----- Fine-tuned Seed {seed} -----")
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

    tvd_scores, outputs, tvd_context = evaluate_model_tvd(model, tokenizer, test_data, device, num_model_samples=40)

    tvd_scores_fine_tuned[seed] = tvd_scores
    model_outputs_fine_tuned[seed] = outputs
    tvd_per_context_fine_tuned[seed] = tvd_context
    mean_val = np.mean(tvd_scores)
    mean_tvd_fine_tuned.append(mean_val)
    print(f"Mean TVD (Fine-tuned) for seed {seed}: {mean_val}")

    # Calculate overlapping outputs for this seed
    overlapping_outputs[seed] = {}
    for context in model_outputs_non_fine_tuned[seed].keys():
        non_ft_outputs = set(model_outputs_non_fine_tuned[seed][context])
        ft_outputs = set(outputs[context])
        overlapping_outputs[seed][context] = list(non_ft_outputs.intersection(ft_outputs))

    with open(f'model_outputs_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in outputs.items():
            f.write(f"{context} -> {preds}\n")

    with open(f'tvd_per_context_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, tvd in tvd_context.items():
            f.write(f"{context}\t{tvd:.6f}\n")

# Save overlapping outputs to JSON
with open('overlapping_outputs.json', 'w') as f:
    json.dump(overlapping_outputs, f, indent=4)

print("\n=== Evaluation Complete. Results saved to respective files ===")

csv_data = []
for seed in seeds:
    row = {
        'Seed': seed,
        'Mean TVD Non-Fine-Tuned': float(np.mean(tvd_scores_non_fine_tuned[seed])),
        'Mean TVD Fine-Tuned': float(np.mean(tvd_scores_fine_tuned[seed])),
        'Full TVD Non-Fine-Tuned': str(tvd_scores_non_fine_tuned[seed]),
        'Full TVD Fine-Tuned': str(tvd_scores_fine_tuned[seed])
    }
    csv_data.append(row)

with open('tvd_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Seed', 'Mean TVD Non-Fine-Tuned', 'Mean TVD Fine-Tuned', 
        'Full TVD Non-Fine-Tuned', 'Full TVD Fine-Tuned'])
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)

print("\nResults saved to tvd_results.csv")

# --- Print comparison ---
print("\n----- Comparison of Fine-tuned vs Non-fine-tuned Models -----")
for seed in seeds:
    non_ft_mean = csv_data[seeds.index(seed)]['Mean TVD Non-Fine-Tuned']
    ft_mean = csv_data[seeds.index(seed)]['Mean TVD Fine-Tuned']
    print(f"Seed {seed}:")
    print(f"  Non-fine-tuned mean TVD: {non_ft_mean}")
    print(f"  Fine-tuned mean TVD: {ft_mean}")
    print(f"  Improvement: {non_ft_mean - ft_mean}")