from evaluation_smart import *
import argparse

path_list =['sep2v3_model_1_40', 'sep2v3_model_1_80', 'sep2v3_model_1_120', 'sep2v3_model_1_160', 'sep2v3_model_1_200', 'sep2v3_model_1']

description = "sep2v3"


def evalutate_trained_models(option, cuda, bs, boot, path_list, description, model_name):
    if boot == "True":
        boot = True
    else:
        boot = False
    evaluation_all(description, option, cuda, bs, path_list, boot, description, model_name)
    
def print_all_evaluation(option, cuda, bs, boot, path_list):
    all_result = []
    all_metric = []
    for path in path_list:
        if 'None' in path:
            continue
        avg_result, metric_dict = print_evaluation(path)
        all_result.append(avg_result)
        all_metric.append(metric_dict)
    print("=" * 50)
    all_keys = all_metric[0].keys()
    for k in all_keys:
        temp = []
        for each in all_metric:
            temp.append(each[k])
        if k[1] != None:
                print(f"Average  :  {k[0]}@{k[1]}  : {np.mean(temp)}%  ||  std: {np.std(temp)}")
        else:
            print(f"checkpoint  :  {k[0]}  : {np.mean(temp)}%  ||  std: {np.std(temp)}")
    print()
    print(f"Average : {np.mean(all_result)}")
    print(f"std : {np.std(all_result)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='benchmark evaluation for Doris Mae',
        epilog='Created by Doris Mae'
    )

    parser.add_argument('-o', '--option', required=True, help='specify whether query option, sub-query or aspect option, format for subquery looks like subquery_k, where k is how many aspects are used, typical is 2')
    parser.add_argument('-c', '--cuda', default= "cpu", help= 'specify cuda ids to be used, format is 1,2,3, or cpu')
    parser.add_argument('-b', '--bs', default = 30, help ='user specified batch size based on their own gpu capability, default is 30, which is tested on GeForce RTX 2080 Titan')
    parser.add_argument('-boot', '--boot',  help ='bootstrap')
    
    
    args = parser.parse_args()
    option = args.option
    cuda = args.cuda.strip()
    bs = int(args.bs)
    if args.boot == "True":
        boot = True
    else:
        boot = False
    
    evaluation_all(description, option, cuda, bs, path_list, boot)
    
