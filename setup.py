import os
import sys
import requests
import shutil
from pathlib import Path

ENTAILMENT_BANK_REPO = 'https://github.com/allenai/entailment_bank.git' 
BLEURT_MODEL_PATH  = 'bleurt-large-512.zip'
BLEURT_MODEL_URL = 'https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip'

ENT_BANK_DATASET_PATH = './entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/'
DATASET_PATH = './data/arc_entail/dataset/'
ENT_BANK_WT_CORPUS_PATH = './entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/supporting_data/'
WT_CORPUS_PATH = './data/arc_entail/supporting_data/'
ENT_BANK_SRC_DATA_PATH = './entailment_bank/data/'
SRC_DATA_PATH = './src/entailment_bank/data/'

BLEURT_FOLDER = './bleurt-large-512'
SRC_BLEURT_FOLDER = './src/entailment_bank/bleurt-large-512'

def prepare_path(path):
    path = Path(path)    
    os.makedirs(path.parents[0], exist_ok=True)
    return path
    
def doanlod_and_save_to_path(url, path):
    path = prepare_path(path)    
    print(f'Downloading:\n{url}')
    response = requests.get(url)
    print(f'Saving to:\n{path}')
    open(path, 'wb').write(response.content)    
    
def copy_folders(from_path, to_path):
    print(f'{from_path} => {to_path}')
    from_path = prepare_path(from_path)
    to_path = prepare_path(to_path)
    shutil.copytree(from_path, to_path, dirs_exist_ok=True)
    
def move_folders(from_path, to_path):
    print(f'{from_path} => {to_path}')
    from_path = prepare_path(from_path)
    to_path = prepare_path(to_path)    
    shutil.move(from_path, to_path)    
    
def clone_git_repo(git_repo):
    # Cloning
    os.system(f'git clone {git_repo}')

def unzip_file(path):
    path = Path(path)
    # Unzipping
    os.system(f'unzip {path}')
    
def setup_entailment_bank_eval():
    print('\nCloning EntailmentBank evaluation repository...')
    clone_git_repo(ENTAILMENT_BANK_REPO)
    print('\nCopying files...')
    copy_folders(ENT_BANK_DATASET_PATH, DATASET_PATH)
    copy_folders(ENT_BANK_WT_CORPUS_PATH,WT_CORPUS_PATH)
    copy_folders(ENT_BANK_SRC_DATA_PATH, SRC_DATA_PATH)

    print('\nDownloading BLEURT model...')
    # downlaad bleurt model
    doanlod_and_save_to_path(BLEURT_MODEL_URL, BLEURT_MODEL_PATH)
    # unzip bleurt model
    unzip_file(BLEURT_MODEL_PATH)
    # copy files to src folder
    print('\nMoving model files...')
    move_folders(BLEURT_FOLDER, SRC_BLEURT_FOLDER)

def main():    
    setup_entailment_bank_eval()
    print('\nSetup finished!')

if __name__ == '__main__':
    main()