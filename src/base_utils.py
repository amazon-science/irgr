import os
import json
import sys
import re

####################################################################
# File Manipulation
####################################################################

def save_to_jsonl_file(output_list, file_path):
    print('saving data to file:', file_path)
    with open(file_path, 'w') as file:
        for obj in output_list:
            file.write(json.dumps(obj) + '\n')
            
def run_funtion_redirect_stdout(fun, args=[], kargs={}, filename='logs.txt'):
    '''
    Runs functions while redirecting output to file 
    '''
    # redirect stdout to file
    orig_stdout = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f
    
    # execute function
    output = fun(*args, **kargs)
    
    # restore stdout
    sys.stdout = orig_stdout
    f.close()
    return output

####################################################################
# Strings
####################################################################

def str_replace_single_pass(string, substitutions):
    '''
    A Python function that does multiple string replace ops in a single pass.
    E.g. substitutions = {"foo": "FOO", "bar": "BAR"}
    '''
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


####################################################################
# Resource Locator
####################################################################

def get_results_file_path(params, test_split=False, result_only=False, 
                          temp=False, uuid=False):    
    suffix = ''
    if test_split:
        suffix += '_test'
    if result_only:
        suffix += '_result_only'
    if temp:
        suffix += '_temp'
    if uuid:
        suffix += '_uuid'
    results_file_path = params.results_file_path.format(
                model_name = params.model_name,
                task_name = params.task_name,
                dataset_name = params.dataset_name,
                approach_name = params.approach_name,
                suffix = suffix,
                extension = 'tsv')
    return results_file_path

def get_logs_file_path(params, test_split=False, temp=False, epoch_num = None):
    suffix = ''
    prefix = ''
    if test_split:
        suffix += '_test'
    if temp:
        suffix += '_temp' 
        prefix += 'temp/'
    if epoch_num is not None:
        suffix += f'_{epoch_num}'
    logs_file_path = params.logs_file_path.format(
        model_name = params.model_name,
        task_name = params.task_name,
        dataset_name = params.dataset_name,
        approach_name = params.approach_name,
        prefix = prefix,
        suffix = suffix
    )
    return logs_file_path

def get_proof_ranking_data_path(params, split = 'dev'):
    logs_file_path = params.proof_ranking_data_filepath.format(
        model_name = params.model_name,
        task_name = params.task_name,
        dataset_name = params.dataset_name,
        approach_name = params.approach_name,   
        split = split
    )
    return logs_file_path