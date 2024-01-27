import pickle
import argparse
import json
import os
import sys
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import time
import shutil
import numpy as np
import pathlib


sys.path.append("./arguana_evaluation")
from beir_eval import *

sys.path.append("./data_prep")
from preparation_large_batch import get_pair_data_gemini, get_pair_data_arguana_gemini
sys.path.append("./training")
from train_large_batch_contrastive_finetuning import training_model_large_batch_contrastive
sys.path.append("./evaluation")
from run_evaluation_smart import evalutate_trained_models, print_all_evaluation
sys.path.append("./wtb_evaluation")
from wtb_eval import evalutate_trained_models_WTB

def find_files_with_prefix(folder_path, prefix):
    file_list = []
    
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if filename.startswith(prefix) and os.path.isfile(full_path):
            file_list.append(filename)
    
    return file_list


def main(query_type="old", num_experiment = 10, name = "vanilla1", query_num = 4000, cuda = "0,1,2,3,4,5,6,7", margin =0.05, batch_size = 80, half= True, shuffle = False, model_name = "e5", max_breach = 5, sent = False):
    

    


    with open(f"./generation/{query_type}.pickle", 'rb') as f:
        query = pickle.load(f)

    if "wtb" in name:
        print("confirming wtb dataset")
        with open("evaluation/dataset/wtb_dataset.pickle", 'rb') as f:
            dataset = pickle.load(f)
    else:
        print("confirming doris mae dataset")
        with open("evaluation/dataset/DORIS_MAE_dataset_v1.json", 'r') as f:
            dataset = json.load(f)

    with open("./evaluation/dataset/arguana_corpus.pickle", 'rb') as f:
        arguana_corpus = pickle.load(f)

    d_fake = [] 
    
    num_cuda = len(cuda.split(","))

    interested_eval_results = []

    all_ret_wtb = []
    all_avg_wtb = []

    for i in range(num_experiment):
        print("==============================Start data generation process==============================")
        experiment_name = f"{name}_{i}_"
        num_epochs = 1

        if 'arguana' in experiment_name:

            
            get_pair_data_arguana_gemini(experiment_name, query_num, query, arguana_corpus, d_fake, shuffle, False, sent)

        else:
        
            get_pair_data_gemini(experiment_name, query_num, query, dataset, d_fake, shuffle, sent)

        print("==============================Start training process==============================")
       
        training_model_large_batch_contrastive(batch_size, num_epochs, experiment_name, cuda, margin, shuffle, half, model_name, max_breach)

        print("==============================Start evaluation process==============================")
        if "wtb" in experiment_name:
            bs = 120
            all_model_path_list = find_files_with_prefix("./training/model_checkpoints", experiment_name)
            assert len(all_model_path_list) == 1
            removed_path = "./training/model_checkpoints/"+ all_model_path_list[0]
            
            path_list = [p.split(".")[0] for p in all_model_path_list]
            ret_wtb, avg_wtb = evalutate_trained_models_WTB(cuda, bs, path_list, experiment_name, model_name)
            all_ret_wtb.append(ret_wtb)
            all_avg_wtb.append(avg_wtb)

        elif "arguana" in experiment_name:
            curr_check_point_path = f"{name}_{i}__model_{model_name}_1"

            ds_name = 'arguana'
            
            with open(f"arguana_evaluation/arguana_result_template.pickle", 'rb') as f:
                curr_result_template = pickle.load(f)
    
            
    
            if name == 'treccovid':
                raise ValueError("Should never happen")
            else:
                dataset = ds_name
                url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
                out_dir = os.path.join(pathlib.Path("./beir_download/").absolute(), "datasets")
                data_path = util.download_and_unzip(url, out_dir)
                #### Provide the data_path where scifact has been downloaded and unzipped
                corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
                
                
                remove_ids = ['test-politics-mtpghwaacb-con02a', 'test-politics-mtpghwaacb-con02b', 'test-international-ghbunhf-con03a', 'test-international-ghbunhf-con03b', 'test-law-cpilhbishioe-con04a', 'test-law-cpilhbishioe-con04b', 'test-education-tuhwastua-pro02a', 'test-education-tuhwastua-pro02b', 'test-law-rmelhrilhbiw-con02a', 'test-law-rmelhrilhbiw-con02b', 'test-economy-epsihbdns-con03a', 'test-economy-epsihbdns-con03b', 'test-politics-oapdhwinkp-con03a', 'test-politics-oapdhwinkp-con03b', 'test-international-gpdwhwcusa-con05a', 'test-international-gpdwhwcusa-con05b', 'test-free-speech-debate-yfsdfkhbwu-con03a', 'test-education-ufsdfkhbwu-con03a', 'test-politics-dhwem-pro06a', 'test-science-sghwbdgmo-con03a', 'test-society-asfhwapg-con04a']
                
        
                new_corpus = corpus.copy()
                new_q = queries.copy()
                new_qrels = qrels.copy()
                for idx in remove_ids:
                    if idx in new_corpus:
                        del new_corpus[idx]
                    if idx in new_q:
                        del new_q[idx]
                    if idx in new_qrels:
                        del new_qrels[idx]
                corpus = new_corpus
                queries = new_q
                qrels = new_qrels
                
                filtered_template = {}
        
                for idx in curr_result_template:
                    if idx not in remove_ids:
                        temp_dict = {}
                        for sub_idx in curr_result_template[idx]:
                            if sub_idx not in remove_ids:
                                temp_dict[sub_idx] = 0.0
                        filtered_template[idx] = temp_dict
        
                curr_result_template = filtered_template
        
                own_model = OwnModel(model_name, curr_check_point_path, ds_name, cuda, 2000)
                reranker= Rerank(own_model, batch_size=128)
                dataset = own_model.get_dataset()
                rerank_results = reranker.rerank(corpus, queries, curr_result_template, top_k=10000)

                eval_result = own_model.evaluate(qrels, rerank_results, [1, 3, 5, 10, 20, 100, 1000, 10000])




                print(eval_result)

    
                
                interested_eval_results.append(eval_result[0]['NDCG@10'])
        else:
            option = "60_query"
            bs = 120
            boot = False
            all_model_path_list = find_files_with_prefix("./training/model_checkpoints", experiment_name)
            assert len(all_model_path_list) == 1
            removed_path = "./training/model_checkpoints/"+ all_model_path_list[0]
            
            path_list = [p.split(".")[0] for p in all_model_path_list]
            evalutate_trained_models(option, cuda, bs, boot, path_list, experiment_name, model_name)

    print("==============================All evaluation result==============================")
    print()
    print()
    print()
    print()
    print("==========================================================================================")
    if 'arguana' in query_type:
        print()
        print()
        print()
        print()
        print("==========================================================================================")
        print(f"{name}_{model_name} averaged NDCG@10: {np.mean(interested_eval_results)}")
        print(f"{name}_{model_name} standard deviation NDCG@10: {np.std(interested_eval_results)}")
        print("Current Experiment Ended")
        print("==========================================================================================")
        print()
        print()
        print()
        print()
    elif 'wtb' in query_type:
        print()
        print()
        print()
        print()
        print("==========================================================================================")
        averages = {}
        for key in all_ret_wtb[0].keys():
            all_ret = [d[key] for d in all_ret_wtb]
            print(f"{name}_{model_name} averaged {key} : {np.mean(all_ret)}, std {key} : {np.std(all_ret)}")
            print()
        print("Current Experiment Ended")
        print("==========================================================================================")
        print()
        print()
        print()
        print()
    else:
        evaluation_path = find_files_with_prefix(f"./evaluation/dataset/rankings/{option}/result", f"result_dict_{name}")
        eval_path_list = sorted([f"./evaluation/dataset/rankings/{option}/result/{p}" for p in evaluation_path])
        print_all_evaluation(option, cuda, bs, boot, eval_path_list)
    print("==========================================================================================")
    print()
    print()
    print()
    print()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="created by Wang")
    
    parser.add_argument("-query_type", "--query_type", type=str, required=True, help="types of query")
    parser.add_argument("-num_experiment", "--num_experiment", type = int, required = True, help = "number of trials")
    parser.add_argument("-name", "--name", type= str, required = True, help = "such as vanilla1")
    parser.add_argument("-query_num", "--query_num", type = int, required = True, help= "number of queries to consider")
    parser.add_argument("-half", "--half", type = str, required = True, help = "whether half freeze")
    parser.add_argument("-shuffle", "--shuffle", type = str, required = True, help = "whether to shuffle")
    parser.add_argument("-cuda", "--cuda", type = str, required = True, help = "list of gpus indices")
    parser.add_argument("-margin", "--margin", type = float, required = True, help = "margin for triplet loss, default 0.05")
    parser.add_argument("-batch_size", "--batch_size", type = int, required = True, help = "total batch size")
    parser.add_argument("-sent", "--sent", type = str, required = True, help = "indicate whether to process it as sentence")
    parser.add_argument("-model_name", "--model_name", type = str, required = True, help = "pretrained model, such as e5")
    args = parser.parse_args()
      
    query_type = args.query_type
    num_experiment = args.num_experiment
    name = args.name
    query_num = args.query_num
    cuda = args.cuda
    margin = args.margin
    model_name = args.model_name
    
    batch_size = args.batch_size
    
    if args.half == "True":
        half = True
    elif args.half == "False":
        half = False
    if args.shuffle == "True":
        shuffle = True
    elif args.shuffle == "False":
        shuffle = False

    if args.sent == "True":
        sent = True
    elif args.sent == "False":
        sent = False
        
        
    main(query_type = query_type, num_experiment = num_experiment, name = name, query_num = query_num, cuda = cuda, margin= margin, batch_size=batch_size, half= half, shuffle = shuffle, model_name = model_name, sent = sent)
    

