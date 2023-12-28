import numpy as np
import pickle
from model_embedding_wtb import load_model_wtb, get_embedding_wtb
import re
import os
import numpy as np
from tqdm import tqdm

def replace_text(paragraph, replace_with=''):
    paragraph = paragraph.replace("•", replace_with)

    paragraph = re.sub(r'<.{1,4}>', replace_with, paragraph)

    paragraph = re.sub(r'&(\w+)', r'\1', paragraph)

    return paragraph

def random_rank(dataset, query, candidate_pool, model_name='random', cuda='cpu', model_path=None):
    return list(np.random.choice(candidate_pool, len(candidate_pool), replace=False))

def compute_all_cosine_similarity(query_dict, candidate_dict):
    query_ids, query_embs = list(query_dict.keys()), np.array(list(query_dict.values()))
    candidate_ids, candidate_embs = list(candidate_dict.keys()), np.array(list(candidate_dict.values()))

    # Compute norms of query and candidate embeddings.
    query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)

    # Compute dot product of queries with all candidates.
    dot_products = np.dot(query_embs, candidate_embs.T)

    # Compute cosine similarities.
    cosine_sims = 1 - dot_products / (query_norms * candidate_norms.T)

    # Convert similarity matrix to dictionary format with query and candidate IDs.
    result = {}
    for i, q_id in enumerate(query_ids):
        result[q_id] = {c_id: cosine_sims[i, j] for j, c_id in enumerate(candidate_ids)}
    return result


def compute_all_L2_distance(query_dict, candidate_dict):
    query_ids, query_embs = list(query_dict.keys()), np.array(list(query_dict.values()))
    candidate_ids, candidate_embs = list(candidate_dict.keys()), np.array(list(candidate_dict.values()))

    result = {}
    for i, q_id in tqdm(enumerate(query_ids)):
        distances = np.linalg.norm(query_embs[i] - candidate_embs, axis=1)
        result[q_id] = {c_id: dist for c_id, dist in zip(candidate_ids, distances)}
    
    return result


def use_model_to_rank(dataset, dataset_name, queries, candidate_pool, model_name, cuda='cpu', batch_size=50, model_path=None):

    # Note query, and corpus are all in 'id' form, the function should use the dataset above to retrieve the text

    if model_path is None:
        file_path = f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_pretrain.pickle"
    else:
        file_path = f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_{model_path}.pickle"
    
    with open(file_path, 'rb') as f:
        id2embedding_dict = pickle.load(f)

    query_emb_dict = {}
    for q in queries:
        query_text = q['query']
        idx = [k for k, v in dataset.items() if v == query_text][0]
        query_emb_dict[idx] = id2embedding_dict[idx]
        
    candidate_emb_dict = {}
    for paper_id in candidate_pool:
        candidate_emb_dict[paper_id] = id2embedding_dict[paper_id]

    distance_dict = {}
    if model_name in ['e5', 'simcse', 'simlm', 'spladev2', 'scibert', 'ernie', 'roberta']:
        # cosine similarity
        distance_dict = compute_all_cosine_similarity(query_emb_dict, candidate_emb_dict)

    elif model_name in ['specterv2']:
        # L2 distance
        distance_dict = compute_all_L2_distance(query_emb_dict, candidate_emb_dict)
    else:
        raise ValueError("Invalid model name")

    sorted_keys_dict = {}
    for q_id in distance_dict:
        sorted_keys_dict[q_id] = sorted(distance_dict[q_id], key=lambda x: distance_dict[q_id][x])
    
    with open(f"./evaluation/embedding_result/{dataset_name}_{model_name}_{model_path}_scores.pickle", 'wb') as f:
        pickle.dump(sorted_keys_dict, f)

    return sorted_keys_dict

def WTB_recall_at_k(dataset, dataset_name, test, k, model_name, candidates_idx, model_path=None):
    if model_name == 'random':
        rank_func = random_rank
    else:
        rank_func = use_model_to_rank

    queries = list(test)
    recall_at_k_list = []
    
    curr_candidate_pool = candidates_idx
    score_path = f"./evaluation/embedding_result/{dataset_name}_{model_name}_{model_path}_scores.pickle"
    if os.path.exists(score_path):
        with open(score_path, 'rb') as f:
            rank_by_model = pickle.load(f)
    else:
        rank_by_model = rank_func(dataset, dataset_name, queries, curr_candidate_pool, model_name=model_name, model_path=model_path)


    for query_and_answer in queries:
        true_label = query_and_answer['unique_id']
        query = query_and_answer['query']
        
        ids = [k for k, v in dataset.items() if v == query]
        if len(ids) == 1:
            q_id = ids[0]
        else:
            raise ValueError('No/multiple ids are found')
        
        first_k_from_model = set(rank_by_model[q_id][:k])
        if true_label in first_k_from_model:
            recall_at_k_list.append(1)
        else:
            recall_at_k_list.append(0)

    return np.mean(recall_at_k_list)

def WTB_embed_text(dataset, dataset_name, model_name, batch_size, cuda='cpu', model_path=None):
    print(f"cuda: {cuda}")
    print("*" * 20 + "Embedding" + "*" * 20)
    print(f"Embedding texts with {model_name}_{model_path}, please be patient")
    text_list = []
    id_text_dict = {}




    for k in list(dataset.keys()):
        text_list.append(replace_text(dataset[k]))
        id_text_dict[k] = replace_text(dataset[k])

    if cuda == 'cpu':
        curr_model, curr_tokenizer = load_model_wtb(model_name, cuda, model_path)
        embedding_result = get_embedding_wtb(model_name, curr_model, curr_tokenizer, text_list, cuda=cuda,
                                         batch_size=batch_size)

    else:
        curr_model, curr_tokenizer = load_model_wtb(model_name, cuda, model_path)
        embedding_result = get_embedding_wtb(model_name, curr_model, curr_tokenizer, text_list, cuda=cuda,
                                         batch_size=batch_size)

    ret = {}
    for paper_id in id_text_dict:
        ret[paper_id] = embedding_result[id_text_dict[paper_id]]

    

    if model_path is None:
        if not os.path.isdir("./evaluation/embedding_result"):
            print("creating dir")
            os.makedirs("./evaluation/embedding_result", exist_ok=True)

        with open(f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_pretrain.pickle", 'wb') as f:
            pickle.dump(ret, f)
    else:

        if not os.path.isdir("./evaluation/embedding_result"):
            print("creating dir")
            os.makedirs("./evaluation/embedding_result", exist_ok=True)

        
        with open(f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_{model_path}.pickle", 'wb') as f:
            pickle.dump(ret, f)


    print(f"Embedding texts with {model_name} finished")

    return None

def evalutate_trained_models_WTB(cuda, bs, path_list, experiment_name, model_name):
    test_suite = {'recall': [5,10,20,100]}
    dataset_name = "wtb"
    model_path = path_list[0]
    
    with open("./evaluation/dataset/wtb_dataset.pickle", 'rb') as f:
        WTB_dataset = pickle.load(f)
    
    data = WTB_dataset['Corpus']
    validation = WTB_dataset['validation']
    test = WTB_dataset['test']
    train = WTB_dataset['train']
    candidatas = WTB_dataset['candidates']
    candidates_idx = list(candidatas.keys())

    dataset = {}
    for k in data:
        dataset[k] = data[k]['title'] + ' . ' + data[k]['description']
        
    for i in range(len(test)):
        dataset[f'q_{i}'] = test[i]['query']

    WTB_embed_text(dataset, dataset_name, model_name, batch_size=bs, cuda=cuda, model_path=model_path)
    
    print("Finish embedding...")

    ret_dict = {}
    for key in test_suite:
        for k in test_suite[key]:
            curr_result = WTB_recall_at_k(dataset, dataset_name, test, k, model_name, candidates_idx, model_path)
            curr_stored_name = f"{key}@{k}__by_{model_name}"
            ret_dict[curr_stored_name] = curr_result

    ret = ret_dict
    max_key_length = max(len(k) for k in ret.keys())
    performance_list = []
    for k in ret:
        performance_list.append(ret[k])
        print(f"{k:<{max_key_length}} : {round(ret[k], 6)}")

    print(f"Averaged: {round(np.mean(performance_list), 6)}")
    return ret, round(np.mean(performance_list), 6)
