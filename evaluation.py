import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoTokenizer,AutoConfig

from train import BarlowTrainingArguments,OurTrainingArguments,VICRegTrainingArguments,ModelArguments
from simcse.model.bert_roberta_models import RobertaForCL, BertForCL, BertForCorInfoMax, RobertaForCorInfoMax,BertForVICReg, RobertaForVICReg,BertForBarlow,RobertaForBarlow
from simcse.model.distilbert_model import DistilBertForCL,DistilBertForCorInfoMax,DistilBertForVICReg,DistilBertForBarlow

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    training_args = torch.load(args.model_name_or_path +"/training_args.bin")
    model_args = torch.load(args.model_name_or_path + "/model_args.bin")
    # Load transformers' model checkpoint
    if("barlow" in args.model_name_or_path):
        if("roberta" in args.model_name_or_path):
            model = RobertaForBarlow(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))


        elif("distilbert" in args.model_name_or_path):
            model = DistilBertForBarlow(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
        else:
            model = BertForBarlow(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
    elif("vicreg" in args.model_name_or_path):
        if("roberta" in args.model_name_or_path):
            model = RobertaForVICReg(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
        elif("distilbert" in args.model_name_or_path):
            model = DistilBertForVICReg(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
        else:
            model = BertForVICReg(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
    else:
        if("roberta" in args.model_name_or_path):
            model = RobertaForCorInfoMax(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
        elif("distilbert" in args.model_name_or_path):
            model = DistilBertForCorInfoMax(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))
        else:
            model = BertForCorInfoMax(config=config,training_args=training_args,model_args = model_args)
            model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.bin"))



    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
            return

    def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(device)
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)

                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()