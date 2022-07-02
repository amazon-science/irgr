# Imports
import os
import sys
import json
import torch
import base_utils
import random
import numpy as np
import pandas as pd
from types import SimpleNamespace
from tqdm.notebook import tqdm

# Importing DL libraries
import torch
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from entailment_bank.eval.run_scorer import main
from retrieval_utils import sent_text_as_counter, convert_datapoint_to_sent_to_text

class CustomDataset(Dataset):

    def __init__(self, source_text, target_text, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        self.source_text = source_text
        self.target_text = target_text

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        source_text = ' '.join(source_text.split())

        target_text = str(self.target_text[index])
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, 
                                                  padding='max_length',return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, 
                                                  padding='max_length',return_tensors='pt', truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

class Trainer():
    '''
    Based on: 
    https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb        
    '''
    def __init__(self, tokenizer=None, model=None, optimizer=None, params = None, config = None):
        self.params = params
        self.config = config
        
        # tokenzier for encoding the text
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.params.model_name)

        # Defining the model.
        self.model = model
        if model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.params.model_name,
            )

        # Defining the optimizer that will be used to tune the weights of 
        # the network in the training session. 
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(params =  self.model.parameters(), 
                                              lr=self.config.LEARNING_RATE)
        
        self.set_random_seed()
        
    def train(self, epoch, train_loader, val_loader, model = None, 
              prefix_constrained_generator = None):
        if model is None:
            model = self.model
        model.train()
        running_loss = 0.0
        tqdm_loader = tqdm(train_loader)        
        for step_num, data in enumerate(tqdm_loader, 0):
            if prefix_constrained_generator is not None:
                prefix_constrained_generator.set_batch_number(step_num)       
            y = data['target_ids'].to(self.params.device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone().detach()
            labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(self.params.device, dtype = torch.long)
            mask = data['source_mask'].to(self.params.device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, 
                                 decoder_input_ids=y_ids, labels=labels)
            loss = outputs[0]

            running_loss += loss.item()            
            if step_num % 100==0:
                avg_loss = running_loss / ((step_num + 1) * self.config.TRAIN_BATCH_SIZE)
                tqdm_loader.set_description("Loss %.4f" % avg_loss)                

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            
        train_avg_loss = running_loss / ((step_num + 1) * self.config.TRAIN_BATCH_SIZE)
        val_avg_loss = self.validate(val_loader, verbose = False)
        print('Epoch: %d, Train Loss:  %.4f, Eval Loss %.4f' % (epoch, train_avg_loss, val_avg_loss))
        return train_avg_loss, val_avg_loss
    
    def validate(self, val_loader, verbose = True, model = None, 
                 prefix_constrained_generator = None):
        if model is None:
            model = self.model
        model.eval()        
        running_loss = 0.0
        tqdm_loader = tqdm(val_loader) if verbose else val_loader
        for step_num, data in enumerate(tqdm_loader, 0):
            if prefix_constrained_generator is not None:
                prefix_constrained_generator.set_batch_number(step_num)
            y = data['target_ids'].to(self.params.device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone().detach()
            labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(self.params.device, dtype = torch.long)
            mask = data['source_mask'].to(self.params.device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, 
                                 decoder_input_ids=y_ids, labels=labels)
            loss = outputs[0]

            running_loss += loss.item()
            if verbose and step_num % 100==0:
                avg_loss = running_loss / ((step_num + 1) * self.config.TRAIN_BATCH_SIZE)
                tqdm_loader.set_description("Loss %.4f" % avg_loss)                

        avg_loss = running_loss / ((step_num + 1) * self.config.TRAIN_BATCH_SIZE)
        if verbose:
            print('Loss: %.4f' % (avg_loss,))
        return avg_loss
    
    
    def predict(self, loader, generation_args = None, model = None, 
                prefix_constrained_generator = None):
        if model is None:
            model = self.model
        model.eval()
        context = []
        predictions = []
        actuals = []
        
        if generation_args is None:
            generation_args = {
                'max_length': self.config.SUMMARY_LEN,
                'num_beams': 3,
                # repetition_penalty': 2.5,
                'length_penalty': 1.0,
                'early_stopping': True
            }
        with torch.no_grad():
            for step_num, data in enumerate(tqdm(loader), 0):
                if prefix_constrained_generator is not None:
                    prefix_constrained_generator.set_batch_number(step_num)
                y = data['target_ids'].to(self.params.device, dtype = torch.long)
                ids = data['source_ids'].to(self.params.device, dtype = torch.long)
                mask = data['source_mask'].to(self.params.device, dtype = torch.long)
                generation_args.update({
                    'input_ids': ids,
                    'attention_mask': mask,
                })
                generated_ids = model.generate(**generation_args)
                inputs = [self.tokenizer.decode(
                    i, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True) for i in ids]
                preds = [self.tokenizer.decode(
                    g, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [self.tokenizer.decode(
                    t, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True) for t in y]
                context.extend(inputs)
                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals, context
    
    def set_random_seed(self):
        # Set random seeds and deterministic pytorch for reproducibility
        torch.manual_seed(self.config.SEED) # pytorch random seed
        np.random.seed(self.config.SEED) # numpy random seed
        torch.backends.cudnn.deterministic = True 
        
    def save_model(self, file_path_suffix = ''):
        model_path = self.params.model_file_path.format(
            model_name = self.params.model_name,
            task_name = self.params.task_name,
            dataset_name = self.params.dataset_name,
            approach_name = self.params.approach_name,
            suffix = file_path_suffix)
        torch.save(self.model.state_dict(), model_path)
        print('state dict saved to: %s' % model_path)

    def load_model(self, file_path_suffix = ''):
        model_path = self.params.model_file_path.format(
            model_name = self.params.model_name,
            task_name = self.params.task_name,
            dataset_name = self.params.dataset_name,
            approach_name = self.params.approach_name,
            suffix = file_path_suffix)
        print('Loading state dict from: %s' % model_path)
        self.model.load_state_dict(torch.load(model_path))

class EntailmentARCDataset():
    
    ROOT_PATH = "../data/arc_entail"
    DATASET_PATH = os.path.join(ROOT_PATH, "dataset")
    TASK_PATH = os.path.join(DATASET_PATH, "task_{task_num}")
    PARTITION_DATA_PATH = os.path.join(TASK_PATH, "{partition}.jsonl")
    
    def __init__(self, semantic_search = None, params = None, config = None):
        self.params = params
        self.config = config
        self.data = {self.get_task_name(task_num): 
                     {partition: [] for partition in ['train', 'dev', 'test']}  
                     for task_num in range(1, 4)}
        self.load_dataset()
        self.semantic_search = semantic_search
    
    def get_task_name(self, task_num):
        return "task_" + str(task_num)
    
    def get_task_number(self, task_name):
        return int(task_name[-1])
    
    def get_dataset_path(self, task_num = 1, partition = 'train'):
        path = self.PARTITION_DATA_PATH.format(task_num = task_num, partition = partition)
        return path
    
    def load_dataset(self):
        for task_name in self.data:
            for partition in self.data[task_name]:
                path = self.get_dataset_path(self.get_task_number(task_name), partition)
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        datapoint = json.loads(line)
                        self.data[task_name][partition].append(datapoint)
    
    def combine_existing_and_search_context(self, datapoint, retrieved):
        # break down existing context to get sent texts
        sent_to_text = convert_datapoint_to_sent_to_text(datapoint)
        # merge retrieved and existing sent texts
        original_ret_size = len(retrieved)
        new_sents = [s['text'] for s in sent_to_text.values()]
        new_sents_lowered = [s['text'].lower() for s in sent_to_text.values()]        
        # add sents retrieved by search to new_sents
        for ret in retrieved:
            if ret.lower() in new_sents_lowered:
                continue
            new_sents.append(ret)
            if len(new_sents) >= original_ret_size:
                break
        assert len(new_sents) <= original_ret_size       
                
        # shuffles order, saving original index
        # new_sent_order[i] == j means original j-th sentence 
        # now in i-th position
        new_sent_order = list(range(len(new_sents)))
        new_context = []
        random.shuffle(new_sent_order)   
        for i in range(len(new_sents)):
            new_context.append(new_sents[new_sent_order[i]])
        # create new contex
        new_context = ' '.join(['sent%d: %s' % (i+1, r) for i, r in enumerate(new_context)])
        # create new proof, modify index according to new context        
        old_to_new_sent_map = {}
        for i in range(len(new_sents)):
            old_to_new_sent_map['sent%d ' % (new_sent_order[i] + 1,)] = 'sent%d ' % (i+1,)
        old_proof = datapoint['proof']
        new_proof = base_utils.str_replace_single_pass(old_proof, old_to_new_sent_map)
        return new_context, new_proof
    
    def update_dataset_with_search(self, dataset, include_existing_context = False):        
        # use retrieved context instead of goden context
        new_dataset = [dict(dp) for dp in dataset]
        retrieved_lst = self.semantic_search.search(
            dataset, top_k = self.params.max_retrieved_sentences)        
        for retrived_it, retrieved in enumerate(retrieved_lst):
            if include_existing_context:
                new_context, new_proof = self.combine_existing_and_search_context(
                    dataset[retrived_it], retrieved)
                if retrived_it < 20:
                    print('OLD CONTEXT = ', dataset[retrived_it]['context'])
                    print('NEW CONTEXT = ', new_context)
                    print('OLD PROOF = ', dataset[retrived_it]['proof'])
                    print('NEW PROOF = ', new_proof)
                    print()
                new_dataset[retrived_it]['context'] = new_context
                new_dataset[retrived_it]['proof'] = new_proof
            else:
                sents = ['sent%d: %s' % (i+1, r) for i, r in enumerate(retrieved)]
                new_dataset[retrived_it]['context'] = ' '.join(sents)
                # makes sure proof is empty since original proof is unrelated to context
                new_dataset[retrived_it]['proof'] = ''
        return new_dataset
        
    def get_source_text(self, task_name, partition):
        source_text = []
        if self.semantic_search is not None:
            new_contexts = self.get_contexts_from_search(
                self.data[task_name][partition])
        for dp_it, data_point in enumerate(self.data[task_name][partition]):
            if self.semantic_search is not None:
                context = new_contexts[dp_it]
            else:
                context = data_point['context']
            hypothesis = data_point['hypothesis']                
            source_text.append(
                'hypothesis: %s, %s' % (hypothesis, context))
        return source_text

    def get_target_text(self, task_name, partition):
        source_text = []
        for data_point in self.data[task_name][partition]:
            source_text.append('$proof$ = %s' % (data_point['proof'],))
        return source_text

    def get_torch_dataloaders(self, task_name, tokenizer):
        '''
        Creation of Dataset and Dataloader for a certain entailment task.
        '''
        # Creating the Training and Validation dataset for further creation of Dataloader
        
        train_source_text = self.get_source_text(task_name, 'train')
        train_target_text = self.get_target_text(task_name, 'train')
        training_set = CustomDataset(train_source_text, train_target_text, tokenizer, 
                                     self.config.MAX_LEN, self.config.SUMMARY_LEN)
        
        dev_source_text = self.get_source_text(task_name, 'dev')
        dev_target_text = self.get_target_text(task_name, 'dev')
        val_set = CustomDataset(dev_source_text, dev_target_text, tokenizer, 
                                self.config.MAX_LEN, self.config.SUMMARY_LEN)

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.config.TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
            }

        val_params = {
            'batch_size': self.config.VALID_BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0
            }

        # Creation of Dataloaders for testing and validation. 
        # This will be used down for training and validation stage for the model.
        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)
        return training_loader, val_loader

class SemanticSearch():
    
    def __init__(self, corpus = None, encoder_model = None, params = None, config = None):
        self.params = params
        self.config = config
        self.encoder_model = encoder_model
        if encoder_model is None:
            self.encoder_model = SentenceTransformer(self.params.sent_trans_name)
        if corpus is not None:
            self.update_corpus(corpus)

    def load_wt_corpus_file(self):
        wt_corpus = {}
        with open(self.params.wt_corpus_file_path, 'r', encoding='utf8') as f:
            wt_corpus = json.loads(f.readline())
        return wt_corpus
            
    def load_wt_corpus(self, extra_facts = None):
        wt_corpus = self.load_wt_corpus_file()
        corpus = list(wt_corpus.values())
        if extra_facts is not None:
            corpus.extend(extra_facts)
            corpus = list(set(corpus))
        self.update_corpus_embeddings(corpus)
    
    def update_corpus_embeddings(self, corpus):
        self.corpus = corpus
        #Encode all sentences in corpus
        self.corpus_embeddings = self.encoder_model.encode(
            corpus, convert_to_tensor=True, show_progress_bar = True)
        self.corpus_embeddings = self.corpus_embeddings.to(self.params.device)
        self.corpus_embeddings = st_util.normalize_embeddings(self.corpus_embeddings)
        
    def search_with_id_and_scores(self, queries, top_k = 1):
        '''
        Search for best semantically similar sentences in corpus.
        
        returns corpus ids (index in input corpus) and scoress
        '''
        if type(queries) != list:
            queries = [queries]

        #Encode all queries
        query_embeddings = self.encoder_model.encode(
            queries, convert_to_tensor=True, show_progress_bar = False)
        query_embeddings = query_embeddings.to(self.params.device)
        query_embeddings = st_util.normalize_embeddings(query_embeddings)
        hits = st_util.semantic_search(query_embeddings, self.corpus_embeddings, 
                                    top_k=top_k, score_function=st_util.dot_score)
        return hits
    
    def search(self, *args, **kwargs):
        '''
        Search for best semantically similar sentences in corpus.
        
        Only returns elements from corpus (no score or id)
        '''
        hits = self.search_with_id_and_scores(*args, **kwargs)
        elements = [[self.corpus[ret['corpus_id']]  for ret in hit] for hit in hits]
        return elements
    
    def run_test(self):
        corpus = [
            'A man is eating food.', 'A man is eating a piece of bread.',
            'The girl is carrying a baby.', 'A man is riding a horse.', 'A woman is playing violin.',
            'Two men pushed carts through the woods.', 'A man is riding a white horse on an enclosed ground.',
            'A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.'
        ]
        self.update_corpus(corpus)
        queries = ['A woman enjoys her meal', 'A primate is performing at a concert']
        results = self.search(queries, top_k = 2)
        for i in range(len(queries)):
            print('Query:', queries[i])
            print('Best results:', results[i])
            print()
    
class PrefixConstrainedGenerator:
    '''
    Constraints the beam search to allowed tokens only at each step. 
    Enforces entailmnet dataset expected format (important for evaluation code)
    '''
    
    def __init__(self, tokenizer, source_text, batch_size):
        # tokenzier for encoding the text
        self.tokenizer = tokenizer
        self.source_text = source_text
        self.batch_size = batch_size
        self.batch_num = 0
    
    def get_first_token_id(self, text):
        toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return toks[0]

    def get_last_token_id(self, text):
        toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return toks[-1]

    def set_batch_number(self, batch_num):
        self.batch_num = batch_num
        
    def set_source_text(self, source_text):
        self.source_text = source_text
    
    def fixed_prefix_allowed_tokens_fn(self, batch_id, inputs_ids):
        '''
        Constrain the next token for beam search depending on currently generated prefix (input_ids)
        The output is loosely formated according to dataset specification.
        '''
        # print(inputs_ids, batch_id)
        prefix = self.tokenizer.decode(inputs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(prefix)
        if prefix.strip() == '':
            return [self.get_first_token_id('sent')]
        if prefix.endswith(' & ') or prefix.endswith(' ; '):
            return [self.get_first_token_id('sent'), self.get_first_token_id('int')]
        if prefix.endswith('sent') or prefix.endswith('int'):
            return [self.get_last_token_id('sent' + str(num)) for num in range(10)]
        if prefix.endswith(' -> '):
            return [self.get_first_token_id('hypothesis'), self.get_first_token_id('int')]
        return list(range(self.tokenizer.vocab_size))
    
    def iterative_prefix_allowed_tokens_fn(self, batch_id, inputs_ids):
        '''
        Constrain the next token for beam search depending on currently generated prefix (input_ids)
        The output is loosely formated according to dataset specification.
        '''
        prefix = self.tokenizer.decode(inputs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        source_idx = self.batch_size * self.batch_num + batch_id        
        source_text = self.source_text[source_idx]
        available_sent_nums = [source_text[match.span()[0] + len('sent'): match.span()[1]-1]
                               for match in re.finditer("(sent)[0-9]+:", source_text)]
        avaliable_int_nums = [source_text[match.span()[0] + len('int'): match.span()[1]-1]
                               for match in re.finditer("(int)[0-9]+:", source_text)]
        
        if prefix.strip() == 'in':
            return [self.get_last_token_id('int')]
        if prefix.strip() == '' or prefix.endswith(' & ') or prefix.endswith(' ; '):
            return [self.get_first_token_id('sent'), self.get_first_token_id('int')]
        if prefix.endswith('sent'):
            return list(set([self.get_last_token_id('sent' + num) for num in available_sent_nums]))
            # return list(set([self.get_last_token_id('sent' + str(num)) for num in range(10)]))
        if prefix.endswith('int') and not prefix.endswith('-> int'):
            return list(set([self.get_last_token_id('int' + num) for num in avaliable_int_nums]))
            # return list(set([self.get_last_token_id('int' + str(num)) for num in range(10)]))
        if prefix.endswith(' -> '):
            return [self.get_first_token_id('hypothesis'), self.get_first_token_id('int')]
        if not ' -> ' in prefix:
            all_toks = list(range(self.tokenizer.vocab_size))
            all_toks.remove(self.get_last_token_id('int1:'))
            return all_toks
        return list(range(self.tokenizer.vocab_size))
    
# Training loop
def run_training_loop(config):
    print('Initiating Fine-Tuning for the model on our dataset')

    min_val_avg_loss = 1e10

    for epoch in range(config.TRAIN_EPOCHS):
        _, val_avg_loss = trainer.train(epoch, training_loader, val_loader)
        if params.save_min_val_loss and val_avg_loss < min_val_avg_loss:
            min_val_avg_loss = val_avg_loss
            # Saving trained model with lowest validation loss
            trainer.save_model(file_path_suffix = '_min_val_loss')

        # Saving trained model
        trainer.save_model()

    print('min_val_avg_loss', min_val_avg_loss)
