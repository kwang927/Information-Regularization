import pickle
import json
import numpy as np
import re
import random
from tqdm import tqdm
import sys
sys.path.append("./evaluation/")
from model_embedding import load_model, get_embedding
from model_scoring import get_relevance_score
import time

def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")


def build_similarity_matrix(texts):
    matrix = np.vstack(list(texts.values()))
    norms = np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    normalized_matrix = matrix / norms
    cosine_sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    
    return cosine_sim_matrix

def get_similarity(text1, text2, texts, sim_matrix):
    # Find indices for the provided texts
    index1 = list(texts.keys()).index(text1)
    index2 = list(texts.keys()).index(text2)
    
    return sim_matrix[index1][index2]

def build_distance_matrix(texts):
    start_time = time.time()
    matrix = np.vstack(list(texts.values()))
    n_texts = len(texts)
    dist_matrix = np.zeros((n_texts, n_texts))
    
    for i in range(n_texts):
        for j in range(n_texts):
            dist = np.linalg.norm(matrix[i] - matrix[j])
            dist_matrix[i][j] = dist
    print(f"build distance matrix need {time.time()-start_time}s")
    return dist_matrix


def check(text1, text2, texts, sim_matrix, l2=False):
    if l2:
        index1 = list(texts.keys()).index(text1)
        index2 = list(texts.keys()).index(text2)
    
        curr_value = sim_matrix[index1][index2].copy()
            
        curr_row = sim_matrix[index1, :].copy()
        
        mask_row = np.zeros_like(curr_row)
        mask_row[index1] = 1000
        mask_row[index2] = 1000
        curr_row = curr_row + mask_row
        
        smallest_in_row = np.min(curr_row)
        if curr_value >= smallest_in_row:
            return False
        else:
            return True

    else:
    
        index1 = list(texts.keys()).index(text1)
        index2 = list(texts.keys()).index(text2)
    
        curr_value = sim_matrix[index1][index2].copy()
            
        curr_row = sim_matrix[index1, :].copy()
        
        mask_row = np.ones_like(curr_row)
        mask_row[index1] = 0
        mask_row[index2] = 0
        curr_row = curr_row * mask_row
        
        largest_in_row = np.max(curr_row)
        if curr_value <= largest_in_row:
            return False
        else:
            return True


def get_consistency_filtered_data(name, model_name, cuda):
       
    model_checkpoint_path = f'{name}_model_{model_name}_1'

    with open(f"data_prep/train_test_data/train_{name}.pickle", "rb") as f:
        train_data = pickle.load(f)
    

    print(f"length of train data triplets {len(train_data)}")

    model, tokenizer = load_model(model_name, cuda, model_checkpoint_path)
    
    text_list = []
    for t in train_data:
        text_list.append(t[0])
        text_list.append(t[1])
        
    text_list = list(set(text_list))
    print(f"total number of documents need embedding {len(text_list)}")

    curr_bs = 1500

    if model_name == 'specterv2':
        curr_bs = 750
    
    l2_flag = False
    embed_dict = get_embedding(model_name, model, tokenizer, text_list, cuda, batch_size = curr_bs)

    if "specter" not in model_name:
        sim_matrix = build_similarity_matrix(embed_dict)
    else:
        l2_flag = True
        sim_matrix = build_distance_matrix(embed_dict)
        
    new_train_data = []
    
    for t in train_data:
        if check(t[0],t[1],embed_dict, sim_matrix, l2_flag):
            new_train_data.append(t)
            
     
    print(f"length of train data triplets after filter {len(new_train_data)}")    
    with open(f"./data_prep/train_test_data/train_{name}.pickle", "wb") as f:
        pickle.dump(new_train_data, f)
  
