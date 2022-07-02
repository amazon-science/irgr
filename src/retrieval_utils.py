import collections
import copy
import json
import re
import rouge
import string
from collections import Counter

def sent_text_as_counter(text):
    text = text.lower().replace(' \'', '\'')
    text = re.sub(r"[.,;/\\?]", "", text)
    tokens = text.split()
    return Counter(tokens)

def create_sentence_uuid_map_from_corpus(world_tree_file):
    uuid_to_sent = {}
    with open(world_tree_file, 'r') as file:
        line = file.readline()
        uuid_to_sent = json.loads(line)        
    sent_txt_to_uuid = []
    num_uuid = 1
    for uuid, wt_text in uuid_to_sent.items():
        sent_txt_to_uuid.append({
            'text': wt_text,
            'text_counter': sent_text_as_counter(wt_text),
            'uuid': uuid,
            'num_uuid': str(num_uuid)
        })
        num_uuid += 1
    return sent_txt_to_uuid

def convert_datapoint_to_sent_to_text(datapoint):
    '''
    creates a mapping from sentences in context to their text
    (e.g. {'sent1': {'text': 'leo is a kind of constellation', 'text_counter': [...]}})
    '''
    context = datapoint['context']    
    matches = list(re.finditer("(sent)[0-9]+:", context))
    sent_to_text = {}
    for match_idx, match in enumerate(matches):
        sent_match = match.group()
        sent_symb = sent_match[:-1] # remove the ':' in "sentX:'
        sent_span = match.span()
        start_pos = sent_span[0] + len(sent_match)
        end_pos = None
        if match_idx + 1 < len(matches):
            end_pos = matches[match_idx + 1].span()[0]
        sent_text = context[start_pos: end_pos].strip()
        sent_to_text[sent_symb] = {
            'text': sent_text,
            'text_counter': sent_text_as_counter(sent_text),
        }
    return sent_to_text

def search_for_sent_uuid(probe, sent_txt_to_uuid):
    '''
    Returns the uuid from worldtree corpus that best match text represented by 
    probe_counter input
    '''
    probe_counter = probe['text_counter']
    best_uuid = None
    best_match = None
    best_match_score = 0
    for wt_item in sent_txt_to_uuid:
        wt_counter = wt_item['text_counter']; wt_uuid = wt_item['num_uuid']
        match_counter = wt_counter & probe_counter
        match_score = sum(match_counter.values())
        if match_score > best_match_score:
            best_uuid = wt_uuid
            best_match = wt_item
            best_match_score = match_score
    ratio = float(best_match_score) / float(sum(probe_counter.values()))
    if ratio < 0.8:
        print('ratio',ratio)
        print('probe', probe)
        print('best_match_counter', best_match)        
        print()
    
    return best_uuid
    
def convert_sentences_num_to_uuid(expression, sent_to_text, sent_txt_to_uuid):
    '''
    Inputs:
    - expression: text containing sentence symbol (e.g. 'sent1 & sen2 -> hypothesis')
    - sent_to_text: dictionary mapping sentence symbols to sentence text
    - sent_txt_to_uuid: dictionary mapping sentence text to worldtree uuid
    '''
    new_expression = expression
    
    # print('sent_to_text', sent_to_text)
    matches = list(re.finditer("(sent)[0-9]+ ", expression))
    for match_idx, match in enumerate(matches):
        sent_symb = match.group()
        if sent_symb[:-1] in sent_to_text:
            sent_text_item  = sent_to_text[sent_symb[:-1]]
            sent_uuid = search_for_sent_uuid(sent_text_item, sent_txt_to_uuid)
        else:
            sent_uuid = 1
        new_expression = new_expression.replace(sent_symb, f"sent{sent_uuid} " )
    return new_expression

def convert_datapoint_sent_to_uuid(datapoint, world_tree_file='data/arc_entail/supporting_data/worldtree_corpus_sentences_extended.json'):
    '''
    converts sentence symbols (e.g. 'sent1') in datapoint text to uuid (e.g. 'sent0239-6af2-d042-caf6')
    '''    
    sent_txt_to_uuid = create_sentence_uuid_map_from_corpus(world_tree_file)
    sent_to_text = convert_datapoint_to_sent_to_text(datapoint)
    new_datapoint = dict(datapoint)
    new_datapoint['proof'] = convert_sentences_num_to_uuid(
        new_datapoint['proof'], sent_to_text, sent_txt_to_uuid)
    return new_datapoint