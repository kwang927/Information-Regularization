import numpy as np
import pickle
from model_embedding_arguana import load_model_arguana, get_embedding_arguana
import os
from tqdm import tqdm

def cosine_similarity_distance(array1, array2):
    dot_product = np.dot(array1, array2)
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)

    if norm1 == 0 or norm2 == 0:
        return 0  # Handle division by zero

    cosine_distance = dot_product / (norm1 * norm2)
    return cosine_distance * 100


def L2_distance(array1, array2):
    return (1 - np.linalg.norm(array1 - array2)) * 100


def replace_text(paragraph, replace_with=''):
#     # Replace • character
#     paragraph = paragraph.replace("•", replace_with)

#     # Replace any text of length less than 5 inside < >
#     paragraph = re.sub(r'<.{1,4}>', replace_with, paragraph)

#     # Replace &some_text with some_text
#     paragraph = re.sub(r'&(\w+)', r'\1', paragraph)

    return paragraph


def get_embed_text(dataset, dataset_name, model_name, batch_size, cuda='cpu', model_path=None):
    print("*" * 20 + "Embedding" + "*" * 20)
    print(f"Embedding texts with {model_name}_{model_path}, please be patient")
    text_list = []
    id_text_dict = {}

    text_to_id_dict = {}

    for k in list(dataset['corpus'].keys()):

        if model_name == 'e5' and dataset_name in ['treccovid', 'scifact']:
            text_list.append(replace_text((dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text'])).strip())
            id_text_dict[k] = (dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text']).strip()
            text_to_id_dict[(dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text']).strip()] = k
        
        else:
            text_list.append(replace_text((dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text']).strip()))
            id_text_dict[k] = replace_text((dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text']).strip())
            text_to_id_dict[(dataset['corpus'][k]['title'] + " " + dataset['corpus'][k]['text']).strip()] = k
        
    for k in list(dataset['query'].keys()):
        if model_name == 'e5' and dataset_name in ['treccovid', 'scifact']:
            text_list.append(replace_text((dataset['query'][k])))
            id_text_dict[k] = replace_text((dataset['query'][k]))
            text_to_id_dict[dataset['query'][k]] = k

        else:
            text_list.append(replace_text(dataset['query'][k]))
            id_text_dict[k] = replace_text(dataset['query'][k])
            text_to_id_dict[dataset['query'][k]] = k
        
    if cuda == 'cpu':
        curr_model, curr_tokenizer = load_model_arguana(model_name, cuda, model_path)
        embedding_result = get_embedding_arguana(model_name, curr_model, curr_tokenizer, text_list, cuda=cuda,
                                         batch_size=batch_size)

    else:
        curr_model, curr_tokenizer = load_model_arguana(model_name, cuda, model_path)
        embedding_result = get_embedding_arguana(model_name, curr_model, curr_tokenizer, text_list, cuda=cuda,
                                         batch_size=batch_size)

    ret = {}
    for paper_id in id_text_dict:
        ret[paper_id] = embedding_result[id_text_dict[paper_id]]
    print(len(id_text_dict))
    print(len(ret))


    store_result = {'text2id_dict': text_to_id_dict, 'embedding_result': ret}



    if model_path is None:
        if not os.path.isdir("./evaluation/embedding_result"):
            print("creating dir")
            os.makedirs("./evaluation/embedding_result", exist_ok=True)
        with open(f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_pretrain.pickle", 'wb') as f:
            pickle.dump(store_result, f)
    else:
        if not os.path.isdir("./evaluation/embedding_result"):
            print("creating dir")
            os.makedirs("./evaluation/embedding_result", exist_ok=True)
        with open(f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_{model_path}.pickle", 'wb') as f:
            pickle.dump(store_result, f)

    print(f"Embedding texts with {model_name} finished")

    return None


def use_model_to_rank(dataset, dataset_name, model_name, cuda='cpu', batch_size=50, model_path=None):


    if model_path is None:
        file_path = f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_pretrain.pickle"
    else:
        file_path = f"./evaluation/embedding_result/{dataset_name}_embedding_result_of_{model_name}_{model_path}.pickle"

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            id2embedding_dict = pickle.load(f)
    else:
        print(f"{file_path} does not exist in the folder, start embedding...")
        get_embed_text(dataset, dataset_name, model_name, batch_size, cuda, model_path)
        with open(file_path, 'rb') as f:
            id2embedding_dict = pickle.load(f)
    
    print("start ranking...")
    all_dict = {}
    for q in tqdm(dataset['query']):
        query_emb = id2embedding_dict[dataset['query'][q]]
        candidate_emb_dict = {}

        for paper_id in dataset['corpus']:
            if model_name == 'e5':
                corpu = replace_text(dataset['corpus'][paper_id]['text'].strip())
                candidate_emb_dict[paper_id] = id2embedding_dict[corpu]
            else:
                corpu = replace_text(dataset['corpus'][paper_id]['text'].strip())
                candidate_emb_dict[paper_id] = id2embedding_dict[corpu]

        distance_dict = {}
        if model_name in ['e5', 'simcse', 'simlm', 'spladev2', 'scibert', 'ernie', 'roberta']:
            for paper_id in candidate_emb_dict:
                distance_dict[paper_id] = cosine_similarity_distance(query_emb, candidate_emb_dict[paper_id])

        elif model_name in ['specterv2']:
            for paper_id in candidate_emb_dict:
                distance_dict[paper_id] = L2_distance(query_emb, candidate_emb_dict[paper_id])
        else:
            raise ValueError("Invalid model name")

        sorted_keys = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
        all_dict[q] = dict(sorted_keys)

    print("finish ranking...")

    return all_dict


def random_rank(dataset, query, candidate_pool, model_name='random', cuda='cpu', model_path=None):
    np.random.seed(42)
    return list(np.random.choice(candidate_pool, len(candidate_pool), replace=False))


