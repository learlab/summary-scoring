import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import logging

import torch
from torch.utils.data import DataLoader

import wandb
from sklearn.metrics import mean_squared_error

from datasets import DatasetDict, concatenate_datasets

from transformers import (Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoTokenizer, AutoModelForSequenceClassification)

assert torch.cuda.is_available(), 'GPU not found. You should fix this.'


def get_args():
    ''' You can run this script with a command like
    python hp_search_vicuna.py --item_id VH271613 --dry_run --model_name_or_path ../bin/vicuna-7b
    '''
    parser = argparse.ArgumentParser()

    # Run Config
    parser.add_argument('--item_id', type=str, default='VH266015', help='Item_id or accession')  # noqa: E501 line too long
    parser.add_argument('--all_items', action='store_true', help='Run all items in the dataset')  # noqa: E501 line too long
    parser.add_argument('--construct_inputs', default=True, action=argparse.BooleanOptionalAction, help='Construct inputs from args.item_prompt_dir. Use --no-construct_inputs to disable.')  # noqa: E501 line too long
    
    # Wandb
    parser.add_argument('--sweep_id', type=str, default=None, help='sweep_id for an existing wandb sweep')  # noqa: E501 line too long
    parser.add_argument('--dry_run', action='store_true', help='Dry run (do not log to wandb)')  # noqa: E501 line too long
    parser.add_argument('--project_name', type=str, default='math-autoscore')
    parser.add_argument('--entity', type=str, default='ai-aloe')

    # Paths
    project_dir = Path(__file__).parent.parent
    parser.add_argument('--question_info_path', type=Path, default=project_dir / 'data' / 'Item_Descriptions_for_NAEP_Math_Scoring_Challenge.csv')  # noqa: E501 line too long
    parser.add_argument('--dataset_path', type=Path, default=project_dir / 'data' / 'naep_data.hf')  # noqa: E501 line too long
    parser.add_argument('--item_prompt_dir', type=Path, default=project_dir / 'vicuna' / 'item_prompts')  # noqa: E501 line too long
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/deberta-v3-large')  # noqa: E501 line too long
    parser.add_argument('--output_dir', type=Path, default=project_dir / 'bin')
    
    # Params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metric', type=str, default='quadratic_kappa')
    parser.add_argument('--maximize', default=True, action=argparse.BooleanOptionalAction, help='Use --no-maximize to minimize the metric.')  # noqa: E501 line too long
    parser.add_argument('--eval_steps', default=1000, type=int)
    parser.add_argument('--save_total_limit', default=4, type=int)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_accumulation_steps', default=2, type=int)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--eval_dataset_size', type=int, default=800, help='Number of samples to use for evaluation per eval dataset')

    # Unused params. Leaving them in for inspiration.
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--no_fp16', action='store_false')
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--no_gradient_checkpointing', action='store_false', default=False)  # noqa: E501 line too long
    parser.add_argument('--log_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=500, type=int)

    return parser.parse_args()


def input_constructor(example, question_info=''):
    '''Constructs an input string for the model.
    '''
    student_response = example['text']

    example['text'] = question_info.replace(
        '<ACTUAL_RESPONSE>',
        student_response
    )

    return example


def tokenize_inputs(example):
    return tokenizer(example['text'], truncation=True)


def load_single_datadict(path, input_constructor_args=None):
    ''' Loads the dataset from disk and preprocess it.'''
    
    datadict = (DatasetDict
                .load_from_disk(path)
                .filter(lambda example: len(example['text'].strip()) > 0)
               )
    
    if input_constructor_args:
        datadict = datadict.map(input_constructor,
                                fn_kwargs=input_constructor_args,
                                desc='Constructing Prompt Inputs')
    
    return datadict.map(tokenize_inputs,
                        batched=True,
                        num_proc=4,
                        remove_columns=['text'],
                        desc='Running tokenizer on dataset')


def load_datadict(args):
    ''' Loads datasets from the disk, reframes them with an input constructor,
    and tokenizes the dicts.
    '''

    train_datasets = []
    dev_datasets = []

    if args.all_items:
        subfolders = [path for path in args.dataset_path.iterdir()
                      if path.is_dir()]
    else:
        subfolders = [args.dataset_path / args.item_id]

    for subfolder in subfolders:
        item_id = subfolder.stem
        
        if args.construct_inputs:
            prompt_path = args.item_prompt_dir / (item_id + '.txt')

            with open(prompt_path) as f:
                input_constructor_args = {
                    'question_info': f.read(),
                }
        else:
            input_constructor_args = None

        temp_datadict = load_single_datadict(subfolder, input_constructor_args)

        train_datasets.append(temp_datadict['train'])
        dev_datasets.append(temp_datadict['dev']
                            .shuffle(seed=args.seed)
                            .select(range(args.eval_dataset_size))
                           )
        
    datadict = {
        'train': concatenate_datasets(train_datasets).shuffle(seed=args.seed),
        'dev': concatenate_datasets(dev_datasets).shuffle(seed=args.seed)
    }

    return datadict


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    mse = mean_squared_error(labels, logits)
    cwk = cohen_kappa_score(labels, np.round(logits), weights='quadratic')

    return {'mse': mse, 'quadratic_kappa': cwk}


def train():
    ''' The main training loop.
    '''
    wandb.init()
    
    config = wandb.config

    model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=1,
            hidden_dropout_prob=config.dropout,
        )
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_total_limit=args.save_total_limit,
        optim='adamw_torch',
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        load_best_model_at_end=False,
        disable_tqdm=False,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datadict['train'],
        eval_dataset=datadict['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == '__main__':

    args = get_args()

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_PROJECT'] = args.project_name
    os.environ['WANDB_ENTITY'] = args.entity
    os.environ['WANDB_LOG_MODEL'] = 'false'
    os.environ['WANDB_DIR'] = args.output_dir.absolute().as_posix()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        truncation_side='left',
    )
    datadict = load_datadict(args)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=16, return_tensors='pt')

    if not args.sweep_id:
        sweep_goal = 'maximize' if args.maximize else 'minimize'
        
        if args.dry_run:
            sweep_name = 'dry-run'
        elif args.all_items:
            sweep_name = f'{args.model_name_or_path}-all-items'
        else:
            sweep_name = f'{args.model_name_or_path}-{args.item_id}'
            
        sweep_config = {
            'name': sweep_name,
            'method': 'bayes',
            'metric': {
                'name': f'eval/{args.metric}',
                'goal': sweep_goal,
            },
            'parameters':
            {
                'epochs': {
                    'values': [2, 3]
                },
                'dropout': {
                    'distribution': 'uniform',
                    'min': 0,
                    'max': 0.2
                },
                'learning_rate': {
                    'distribution': 'uniform',
                    'min': 1e-5,
                    'max': 2e-5,
                },
                'weight_decay': {
                    'values': [0.3]
                },
            },
        }

        sweep_id = wandb.sweep(sweep_config,
                               entity=args.entity,
                               project=args.project_name)

    else:
        sweep_id = args.sweep_id
        
    wandb.agent(sweep_id, train, count=20)
