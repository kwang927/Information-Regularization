from dataset_util import *
from rank_performance_util import get_metric, get_random_baseline, print_random_baseline, get_full_metric_results
from config import *
import json
from model_scoring import *
import argparse
import numpy as np
import os

def remove_one_file(file_path):
    # The helper funciton to remove one file
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while trying to remove the file: {e}")

def remove_files(rank_file_path, embed_file_path):
    # To perform new embeddings of models, we need to first remove the old ones with same name
    remove_one_file(rank_file_path)
    remove_one_file(embed_file_path)


def evaluation_all(description, option, cuda, bs, path_list, bootstrap=True, experiment_name = None, model_name="e5"):
    # To run the evaluation on all models in the path_list
    
    bs = int(bs)
    test_suites = {'recall': [5,20], 'r_precision':[None], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}
    
    if option not in ["query", "aspect", "10_candidate", "first_query", "ten_query", "60_query"]:
        assert "subquery" in option.split("_"), f"option format error {option}"
        
        
    print("loading dataset.... may take a while")
    with open("./evaluation/dataset/DORIS_MAE_dataset_v1.json", "r") as f:
        dataset = json.load(f)
        print(f"raw dataset size {len(dataset['Query'])}")
    
    
    if "subquery" in option:
        print(f"creating {option} dataset....")
        num_aspect = int(option.split("_")[-1])
        dataset = create_subquery_dataset(dataset, num_aspect = num_aspect)
        print(f"{option} dataset size {len(dataset['Query'])}")
    elif option == "aspect":
        print(f"creating aspect dataset...")
        dataset = create_aspect_dataset(dataset)
#         dataset = create_individual_aspect_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
    
    elif option == "10_candidate":
        print(f"creating 10 candidate dataset...")
#         dataset = create_10_candidate_dataset(dataset)
        list_to_follow = [2, 8,13,17,24,26,35,37,41,49]
        dataset = create_10_candidate_dataset_sub(dataset, list_to_follow)
        
        print(f"{option} dataset size {len(dataset['Query'])}")
    elif option == "first_query":
        print(f"creating first query dataset...")
        dataset = create_first_query_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
        
    elif option == "ten_query":
        print(f"creating ten query dataset...")
        dataset = create_ten_query_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
    
    elif option == "60_query":
        print(f"creating 60 query dataset...")
        dataset = create_60_query_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
        
    gpt_result = compute_all_gpt_score(dataset)
    config = create_config( option,cuda,  bs)
    
    
    level = config["level"]
    
    if model_name not in ["e5", "simcse", "roberta", "specterv2", "simlm", "spladev2", "scibert", "ernie"]:
        raise ValueError("Invalid model name")

    query_mode = config["model_name_dict"][model_name]["query_mode"]
    abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
    aggregation = config["model_name_dict"][model_name]["aggregation"]
    
    rank_file_path = f"{rank_result_path}/{level}/ranking_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle"
    embed_file_path = f"{embedding_result_path}/{level}/embedding_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle"
    
    
    
            
    final_result_dict={}
    
    my_model = model_name
    
    print(f"model name : {my_model}")
    print("-"*65)
    for e5v3_path in path_list:
        remove_files(rank_file_path, embed_file_path)
        print(f"begin evaluating model path {e5v3_path}")
        if bootstrap:
            result_dict = evaluation_bootstrap(option, cuda, bs, e5v3_path, config, gpt_result, dataset, my_model)
        else:
            result_dict = evaluation(option, cuda, bs, e5v3_path, config, gpt_result, dataset, my_model)
            
        final_result_dict[e5v3_path] = result_dict
        
    print("Embedding done")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print(f"model name : {my_model}")
    print("-"*65)
    if bootstrap:
        for e5v3_path in path_list: 
            result_dict = final_result_dict[e5v3_path]
            for test_name in result_dict:
                ave = np.mean(result_dict[test_name])
                std = np.std(result_dict[test_name])
                if test_name[1] != None:
                        print(f"checkpoint: {e5v3_path}  : {test_name[0]}@{test_name[1]}  :  {ave:.5f} +- {std:.5f}%")
                else:
                    print(f"checkpoint: {e5v3_path}  : {test_name[0]}  :  {ave:.5f} +- {std:.5f}%")
            print("-"*65)
        
    else:
        for e5v3_path in path_list: 
            result_dict = final_result_dict[e5v3_path]
            all_value = []
            for test_name in result_dict:
                value = np.mean(result_dict[test_name]) * 100
                all_value.append(value)
                if test_name[1] != None:
                        print(f"checkpoint: {e5v3_path}  :  {test_name[0]}@{test_name[1]}  : {value}%")
                else:
                    print(f"checkpoint: {e5v3_path}  :  {test_name[0]}  : {value}%")
            print(f"checkpoint: {e5v3_path} :  Average metrics  :  {np.mean(all_value)}%")
            print("-"*65)

    if not os.path.isdir(f"{rank_result_path}/{level}/result"):
            print("creating dir")
            os.makedirs(f"{rank_result_path}/{level}/result", exist_ok=True)
    with open(f"{rank_result_path}/{level}/result/result_dict_{experiment_name}.pickle", "wb") as f:
        pickle.dump(final_result_dict, f)

def print_evaluation(rank_path):
    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5,20], 'r_precision':[None], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}
    
    with open(rank_path, 'rb') as f:
        result_dict = pickle.load(f)

    all_value = []
    metric_dict = {}
    checkpoint_name = list(result_dict.keys())[0]
    for test_name in result_dict[checkpoint_name]:
        value = round(np.mean(result_dict[checkpoint_name][test_name]) * 100, 3)
        all_value.append(value)
        metric_dict[test_name] = value
        if test_name[1] != None:
                print(f"checkpoint: {checkpoint_name}  :  {test_name[0]}@{test_name[1]}  : {value}%")
        else:
            print(f"checkpoint: {checkpoint_name}  :  {test_name[0]}  : {value}%")
    print(f"checkpoint: {checkpoint_name} :  Average metrics  :  {np.mean(all_value)}%")
    print("-"*65)

    return np.mean(all_value), metric_dict


def evaluation(option, cuda, bs, e5v3_path, config, gpt_result, dataset, my_model):

    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5,20], 'r_precision':[None], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}
    
    
    
    level = config["level"]
    
    model_name = my_model

    
    query_mode = config["model_name_dict"][model_name]["query_mode"]
    abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
    aggregation = config["model_name_dict"][model_name]["aggregation"]
    rank = rank_by_model(dataset, model_name, config, e5v3_path)

    result_dict = {}
    for test_name in test_suites.keys():
        for k in test_suites[test_name]:    
            ret_list = get_full_metric_results(gpt_result, rank, test_name, k)
            result_dict[(test_name, k)] = ret_list
        
            
    return result_dict
        
    

def evaluation_bootstrap(option, cuda, bs, e5v3_path, config, gpt_result, dataset, my_model):
        
    
    
    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5,20], 'r_precision':[None], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}
    
    
    
    level = config["level"]

    model_name = my_model

    
    query_mode = config["model_name_dict"][model_name]["query_mode"]
    abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
    aggregation = config["model_name_dict"][model_name]["aggregation"]
    rank = rank_by_model(dataset, model_name, config, e5v3_path)
    
    
    result_dict = {}

    
    for _ in tqdm(range(1000)):
        boot_indices = np.random.choice(range(len(rank)), len(rank), replace =True)
        new_rank = [rank[idx] for idx in boot_indices]
        new_gpt_result = [gpt_result[idx] for idx in boot_indices]

        for test_name in test_suites.keys():
            for k in test_suites[test_name]:    
                ret_list = get_full_metric_results(new_gpt_result, new_rank, test_name, k)
                if (test_name, k) not in result_dict:
                    result_dict[(test_name, k)] = []
                result_dict[(test_name, k)].append(np.mean(ret_list)*100)
        
            
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='benchmark evaluation for Doris Mae',
        epilog='Created by Doris Mae'
    )

    parser.add_argument('-o', '--option', required=True, help='specify whether query option, sub-query or aspect option, format for subquery looks like subquery_k, where k is how many aspects are used, typical is 2')
    parser.add_argument('-c', '--cuda', default= "cpu", help= 'specify cuda ids to be used, format is 1,2,3, or cpu')
    parser.add_argument('-b', '--bs', default = 30, help ='user specified batch size based on their own gpu capability, default is 30, which is tested on GeForce RTX 2080 Titan')
    parser.add_argument('-p', '--e5_path', required=True, help ='path of e5v3')
    
    args = parser.parse_args()
    option = args.option
    cuda = args.cuda.strip()
    bs = int(args.bs)
    e5v3_path = args.e5_path
    
    if option not in ["query", "aspect", "10_candidate", "first_query", "ten_query"]:
        assert "subquery" in option.split("_"), f"option format error {option}"
    
    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5], 'r_precision':[None], 'ndcg':[0.1], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None], "binary_acc":[None]}


    print("loading dataset.... may take a while")
    with open("dataset/DORIS_MAE_dataset_v0.json", "r") as f:
            dataset = json.load(f)
            print(f"raw dataset size {len(dataset['Query'])}")
            
    if "subquery" in option:
        print(f"creating {option} dataset....")
        num_aspect = int(option.split("_")[-1])
        dataset = create_subquery_dataset(dataset, num_aspect = num_aspect)
        print(f"{option} dataset size {len(dataset['Query'])}")
    elif option == "aspect":
        print(f"creating aspect dataset...")
        dataset = create_aspect_dataset(dataset)
#         dataset = create_individual_aspect_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
    
    elif option == "10_candidate":
        print(f"creating 10 candidate dataset...")
#         dataset = create_10_candidate_dataset(dataset)
        list_to_follow = [2, 8,13,17,24,26,35,37,41,49]
        dataset = create_10_candidate_dataset_sub(dataset, list_to_follow)
        
        print(f"{option} dataset size {len(dataset['Query'])}")
    elif option == "first_query":
        print(f"creating first query dataset...")
        dataset = create_first_query_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")
        
    elif option == "ten_query":
        print(f"creating ten query dataset...")
        dataset = create_ten_query_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")

    # calculate ground truth ranking for the candidate pool of each query, as provided by GPT
    gpt_result = compute_all_gpt_score(dataset)

    # create a configuration dictionary
    config = create_config( option,cuda,  bs)
    
    level = config["level"]
    if os.path.exists(f"{rank_result_path}/{level}/ranking_random.pickle"):
        with open(f"{rank_result_path}/{level}/ranking_random.pickle", "rb") as f:
            random_result_dict = pickle.load(f)
        print_random_baseline(test_suites, random_result_dict)
    else:
        model_name = "ernie"
        rank = rank_by_model(dataset, model_name, config)
        random_result_dict = get_random_baseline(gpt_result, rank, test_suites, trials = 100)
        print_random_baseline(test_suites, random_result_dict)
        with open(f"{rank_result_path}/{level}/ranking_random.pickle", "wb") as f:
            pickle.dump(random_result_dict, f)

    
    for model_name in config["model_name_dict"].keys():
        query_mode = config["model_name_dict"][model_name]["query_mode"]
        abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
        aggregation = config["model_name_dict"][model_name]["aggregation"]
        rank = rank_by_model(dataset, model_name, config, e5v3_path)
        for test_name in test_suites.keys():
            for k in test_suites[test_name]:    
                value = get_metric(gpt_result, rank, test_name, k)
                if k != None:
                    print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}@{k}  : {value}%")
                else:
                    print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}  : {value}%")

        print("-"*75+"\n")    

    
