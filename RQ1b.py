import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import numpy as np
import random

from huggingface_hub import login
# login("VALID LOGIN CREDENTIALS HERE")

seeds = [42, 123, 456]
tvd_scores_fine_tuned = {}
mean_tvd_fine_tuned = []
for seed in seeds:
    print(f"----- Starting Seed {seed} -----")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multiple GPUs, though torch.manual_seed often covers this
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'left'

    train_dataset = load_dataset('json', data_files='train_dataset.json', split='train')
    val_dataset = load_dataset('json', data_files='val_dataset.json', split='train')

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda:0", quantization_config=bnb_config
    )

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
        ## GROUP 1: Memory usage
        # These arguments will squeeze the most out of your GPU's RAM

        # Checkpointing
        gradient_checkpointing=True,    # this saves a LOT of memory
        # Set this to avoid exceptions in newer versions of PyTorch
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # Gradient Accumulation / Batch size
        # Actual batch (for updating) is same (1x) as micro-batch size
        gradient_accumulation_steps=1,
        # The initial (micro) batch size to start off with
        per_device_train_batch_size=32,
        # If batch size would cause OOM, halves its size until it works
        auto_find_batch_size=True,

        ## GROUP 2: Dataset-related
        max_seq_length=96,
        # Dataset
        # packing a dataset means no padding is needed
        packing=True,

        ## GROUP 3: These are typical training parameters
        num_train_epochs=4,
        learning_rate=1e-4,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim='paged_adamw_8bit',

        ## GROUP 4: Logging parameters
        logging_steps=100,
        logging_dir='./logs',
        output_dir=f'./multilabel_finetuned_mistral7b_seed_{seed}',
        report_to='none',
        # evaluation_strategy= 'epoch'
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset
    )

    trainer.train()

    trainer.save_model(f"./fine_tuned_mistral7b_seed_{seed}")
