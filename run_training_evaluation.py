import pickle
import argparse
import json
import os
import sys
import pathlib
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import numpy as np
sys.path.append("./arguana_evaluation")
from beir_eval import *
from data_prep.preparation_large_batch import get_pair_data, get_pair_data_arguana
from training.train_large_batch import training_model_large_batch
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


def main(query_type="old", num_experiment = 10, name = "vanilla1", query_num = 4000, cuda = "0,1,2,3,4,5,6,7", margin =0.05, batch_size = 80, half= True, shuffle = False, model_name = "e5", max_breach = 5, evaluation = True):
    
    print(f"Experiment setting: query type: {query_type}, num_experiment: {num_experiment}, name: {name}, query_num: {query_num}, half freeze: {half},  shuffle data: {shuffle}, cuda :{cuda}, margin: {margin}, max breach for early stopping {max_breach}, total batch size: {batch_size} ")
    print()
    


    with open(f"./generation/{query_type}.pickle", 'rb') as f:
        query = pickle.load(f)

    if "wtb" in name:
        print("confirming wtb dataset")
        with open("evaluation/dataset/wtb_dataset.pickle", 'rb') as f:
            dataset = pickle.load(f)
    else:
        print("confirming doris mae or Arguana dataset")
        with open("evaluation/dataset/DORIS_MAE_dataset_v1.json", 'r') as f:
            dataset = json.load(f)

    with open("./evaluation/dataset/arguana_corpus.pickle", 'rb') as f:
        arguana_corpus = pickle.load(f)

    d_fake = [] 
    ret_list = []
    avg_list = []

    for i in range(num_experiment):
        print("==============================Start data generation process==============================")
        experiment_name = f"{name}_{i}_"
        num_epochs = 1

        if 'arguana' in experiment_name:

            print("For Arguana")

            if 'Qreg' in experiment_name:
                get_pair_data_arguana(experiment_name, query_num, query, arguana_corpus, d_fake, shuffle, breakdown = True)
            else:
                get_pair_data_arguana(experiment_name, query_num, query, arguana_corpus, d_fake, shuffle, breakdown = False)

        else:
        
            get_pair_data(experiment_name, query_num, query, dataset, d_fake, shuffle)

        print("==============================Start training process==============================")
       
        training_model_large_batch(batch_size, num_epochs, experiment_name, cuda, margin, shuffle, half, model_name, max_breach)

        if evaluation:

            print("==============================Start evaluation process==============================")

            bs = 4096
            all_model_path_list = find_files_with_prefix("./training/model_checkpoints", experiment_name)
            # assert len(all_model_path_list) == 1
            
            path_list = [p.split(".")[0] for p in all_model_path_list]

            if 'wtb' in experiment_name:
                ret_wtb, avg_wtb = evalutate_trained_models_WTB(cuda, bs, path_list, experiment_name, model_name)
                ret_list.append(ret_wtb)
                avg_list.append(avg_wtb)
            elif 'arguana' in experiment_name:
                curr_check_point_path = f"{name}_{i}__model_{model_name}_1"

                ds_name = 'arguana'
                
                with open(f"./evaluation/arguana_result_template.pickle", 'rb') as f:
                    curr_result_template = pickle.load(f)
        

                dataset = ds_name
                url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
                out_dir = os.path.join(pathlib.Path("./beir_download/").absolute(), "datasets")
                data_path = util.download_and_unzip(url, out_dir)
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
                eval_result = own_model.evaluate(qrels, rerank_results, [1, 3, 5, 10, 100, 1000])
                print(eval_result)
                ret_list.append(eval_result[0]['NDCG@10'])


            elif "doris-mae" in experiment_name:
                option = "60_query"
                bs = 120
                boot = False
                all_model_path_list = find_files_with_prefix("./training/model_checkpoints", experiment_name)
                assert len(all_model_path_list) == 1            
                path_list = [p.split(".")[0] for p in all_model_path_list]
                evalutate_trained_models(option, cuda, bs, boot, path_list, experiment_name, model_name)
            else:
                raise RuntimeError("You can write customized evaluation here")


    if evaluation:
        print("==============================All evaluation result==============================")
        if 'wtb' in experiment_name:
            print()
            print()
            print()
            print()
            print("==========================================================================================")
            for key in ret_list[0].keys():
                all_ret = [d[key] for d in ret_list]
                print(f"{name}_{model_name} averaged {key} : {np.mean(all_ret)}, std {key} : {np.std(all_ret)}")
                print()
            print("Current Experiment Ended")
            print("==========================================================================================")
            print()
            print()
            print()
            print()
        elif 'arguana' in experiment_name:
            print()
            print()
            print()
            print()
            print("==========================================================================================")
            print(f"{name}_{model_name} averaged NDCG@10: {np.mean(ret_list)}")
            print(f"{name}_{model_name} standard deviation NDCG@10: {np.std(ret_list)}")
            print("Current Experiment Ended")
            print("==========================================================================================")
            print()
            print()
            print()
            print()
        elif 'doris-mae' in experiment_name:
            evaluation_path = find_files_with_prefix(f"./evaluation/dataset/rankings/{option}/result", f"result_dict_{name}")
            eval_path_list = sorted([f"./evaluation/dataset/rankings/{option}/result/{p}" for p in evaluation_path])
            print_all_evaluation(option, cuda, bs, boot, eval_path_list)
        else:
            raise RuntimeError("You can write customized evaluation here")
    else:
        print("Program Finished")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("-query_type", "--query_type", type=str, required=True, help="types of query")
    parser.add_argument("-num_experiment", "--num_experiment", type = int, required = True, help = "number of trials")
    parser.add_argument("-name", "--name", type= str, required = True, help = "such as vanilla1")
    parser.add_argument("-query_num", "--query_num", type = int, required = True, help= "number of queries to consider")
    parser.add_argument("-half", "--half", type = str, required = True, help = "whether half freeze")
    parser.add_argument("-shuffle", "--shuffle", type = str, required = True, help = "whether to shuffle")
    parser.add_argument("-cuda", "--cuda", type = str, required = True, help = "list of gpus indices")
    parser.add_argument("-margin", "--margin", type = float, required = True, help = "margin for triplet loss, default 0.05")
    parser.add_argument("-batch_size", "--batch_size", type = int, required = True, help = "total batch size")
    # parser.add_argument("-max_breach", "--max_breach", type = int, required = True, help = "for early stopping")
    parser.add_argument("-model_name", "--model_name", type = str, required = True, help = "pretrained model, such as e5")
    parser.add_argument("-evaluation", "--evaluation", type = str, required = True, help = "whether run evaluation")

    args = parser.parse_args()
    
    query_type = args.query_type
    num_experiment = args.num_experiment
    name = args.name
    query_num = args.query_num
    cuda = args.cuda
    margin = args.margin
    model_name = args.model_name
    # max_breach = args.max_breach
    
    batch_size = args.batch_size
    evaluation = args.evaluation.lower() == 'true'
    
    if args.half == "True":
        half = True
    elif args.half == "False":
        half = False
    if args.shuffle == "True":
        shuffle = True
    elif args.shuffle == "False":
        shuffle = False

    main(query_type = query_type, num_experiment = num_experiment, name = name, query_num = query_num, cuda = cuda, margin= margin, batch_size=batch_size, half= half, shuffle = shuffle, model_name = model_name, evaluation = evaluation)
    
    # example:
    # nohup python3 -u run_training_evaluation.py -query_type old -num_experiment 10 -name vanilla1 -query_num 4000 -half True -shuffle False > experimental_loggings/experiment_vanilla1_reproduce.txt 2>&1 &

