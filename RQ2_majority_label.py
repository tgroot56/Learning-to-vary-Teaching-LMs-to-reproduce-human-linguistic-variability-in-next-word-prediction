import os
import torch
import random
import numpy as np
import re
import json
import csv
from collections import defaultdict, Counter
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from huggingface_hub import login
# login("VALID LOGIN CREDENTIALS HERE")

# === Seeding ===
seeds = [42, 123, 456]

for seed in seeds:
    print(f"\n----- Fine-tuning Seed {seed} -----")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'left'

    train_dataset = load_dataset('json', data_files='train_dataset_majority_label.json', split='train')
    # val_dataset = load_dataset('json', data_files='val_dataset.json', split='train')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=1,
        per_device_train_batch_size=32,
        auto_find_batch_size=True,
        max_seq_length=96,
        packing=True,
        num_train_epochs=4,
        learning_rate=1e-4,
        optim='paged_adamw_8bit',
        logging_steps=100,
        logging_dir='./logs',
        output_dir=f'./majority_label_multilabel_finetuned_mistral7b_seed_{seed}',
        report_to='none'
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(f"./majority_label_fine_tuned_mistral7b_seed_{seed}")

# === Evaluation Script ===
from peft import PeftModel
from collections import defaultdict, Counter
from datasets import load_dataset

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def extract_word_after_context(reply, context):
    reply_clean = clean_text(reply)
    context_clean = clean_text(context)
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

def group_by_context(test_data):
    grouped = defaultdict(list)
    for sample in test_data:
        context = sample['messages'][1]['content']
        next_word = sample['messages'][2]['content'].strip().lower()
        grouped[context].append(next_word)
    return grouped

def get_estimator(elements):
    c = Counter(elements)
    support = list(c.keys())
    counts = list(c.values())
    probs = [count / sum(counts) for count in counts]
    return support, probs

def get_common_support(s1, s2): return set(s1).union(set(s2))

def change_support(old_supp, old_probs, new_supp):
    new_probs = []
    for item in new_supp:
        if item in old_supp:
            ind = old_supp.index(item)
            new_probs.append(old_probs[ind])
        else:
            new_probs.append(0)
    return list(new_supp), new_probs

def get_tvd(p1, p2):
    return np.sum(np.abs(np.array(p1) - np.array(p2))) / 2

def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    return tokenizer.apply_chat_template(converted_sample, tokenize=False, add_generation_prompt=True)

def get_first_word_sample(model, tokenizer, contexts, add_tokens=5, device=None):
    system_prompt = "Only return one plausible next word for the following the sentence."
    messages = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}] for context in contexts]
    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, do_sample=True,
                             num_beams=1, num_return_sequences=1, max_new_tokens=add_tokens,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [extract_word_after_context(d.split("[/INST]")[-1].strip(), c).lower() for d, c in zip(decoded, contexts)]

def evaluate_model_tvd(model, tokenizer, test_data, device, num_model_samples=40):
    tvd_scores, model_outputs, tvd_per_context = [], {}, {}
    grouped_data = group_by_context(test_data)
    for context, human_words in grouped_data.items():
        contexts_batch = [context] * num_model_samples
        model_preds = get_first_word_sample(model, tokenizer, contexts_batch, device=device)
        model_preds = [pred for pred in model_preds if pred]
        if context not in model_outputs:
            model_outputs[context] = []
        model_outputs[context].extend(model_preds)
        s_h, p_h = get_estimator(human_words)
        s_m, p_m = get_estimator(model_preds)
        union_supp = get_common_support(s_h, s_m)
        s_h, p_h = change_support(s_h, p_h, union_supp)
        s_m, p_m = change_support(s_m, p_m, union_supp)
        tvd = get_tvd(p_h, p_m)
        tvd_scores.append(tvd)
        tvd_per_context[context] = tvd
    return tvd_scores, model_outputs, tvd_per_context

# --- Evaluation Main ---
print("\n=== Starting Evaluation ===")
test_data = load_dataset('json', data_files='test_dataset.json', split='train')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
base_model.eval()

tvd_scores_non_fine_tuned, tvd_scores_fine_tuned = {}, {}
mean_tvd_non_fine_tuned, mean_tvd_fine_tuned = [], []

# --- Evaluate Base Model ---
for seed in seeds:
    print(f"Evaluating base model - Seed {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    tvd, outputs, _ = evaluate_model_tvd(base_model, tokenizer, test_data, device)
    tvd_scores_non_fine_tuned[seed] = tvd
    mean_tvd_non_fine_tuned.append(np.mean(tvd))
    print(f"Base Model TVD (seed {seed}): {np.mean(tvd)}")
    with open(f'majority_label_model_outputs_non_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in outputs.items():
            f.write(f"{context} -> {preds}\n")

# --- Evaluate Fine-tuned Models ---
for seed in seeds:
    print(f"Evaluating fine-tuned model - Seed {seed}")
    model_path = f"./majority_label_fine_tuned_mistral7b_seed_{seed}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = PeftModel.from_pretrained(model, model_path, config=lora_config).to(device)
    model.eval()
    tvd, outputs, _ = evaluate_model_tvd(model, tokenizer, test_data, device)
    tvd_scores_fine_tuned[seed] = tvd
    mean_tvd_fine_tuned.append(np.mean(tvd))
    print(f"Fine-tuned Model TVD (seed {seed}): {np.mean(tvd)}")

    with open(f'majority_label_model_outputs_fine_tuned_seed{seed}.txt', 'w') as f:
        for context, preds in outputs.items():
            f.write(f"{context} -> {preds}\n")

# --- Save Results ---
with open('majority_label_tvd_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Seed', 'Mean TVD Non-Fine-Tuned', 'Mean TVD Fine-Tuned', 'Full TVD Non-Fine-Tuned', 'Full TVD Fine-Tuned'
    ])
    writer.writeheader()
    for seed in seeds:
        writer.writerow({
            'Seed': seed,
            'Mean TVD Non-Fine-Tuned': np.mean(tvd_scores_non_fine_tuned[seed]),
            'Mean TVD Fine-Tuned': np.mean(tvd_scores_fine_tuned[seed]),
            'Full TVD Non-Fine-Tuned': str(tvd_scores_non_fine_tuned[seed]),
            'Full TVD Fine-Tuned': str(tvd_scores_fine_tuned[seed])
        })

print("\n=== Evaluation Complete. Results saved to tvd_results.csv ===")