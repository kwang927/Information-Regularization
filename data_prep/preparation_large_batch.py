import pickle
import json
import numpy as np
import re
import random
from tqdm import tqdm
import sys
import numpy as np

def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")     

def get_pair_large_batch(keys, query_dict, dataset):
    pairs = [ ]
    for k in keys:
        
        D = (preprocessing(query_dict[k]['query']), preprocessing(dataset['Corpus'][k]['original_abstract']))
        pairs.append([D])

    return pairs


def get_pair_large_batch_wtb(keys, query_dict, dataset):
    pairs = [ ]
    for k in keys:
        text = dataset['Corpus'][k]['title']+ " . "+ dataset['Corpus'][k]['description']
        
        D = (preprocessing(query_dict[k]['query']), preprocessing(text))
        pairs.append([D])

    return pairs


def get_pair_data_arguana(experiment_name, query_num, query, arguana_corpus, d_fake, shuffle, breakdown = False):

    if breakdown:

        if 'breakdown' not in experiment_name:
            raise ValueError("Should not happen for breakdown preparation")

        query_dict = {}
        for q in query:
            query_dict[q['id']] = {'query': q['query'], 'breakdown': q['breakdown']}
    
        keys = list(query_dict.keys())[:query_num]
    
        if 'arguana' not in experiment_name:
            raise ValueError("Experiment_name has no Arguana")
    
        pairs = []
    
        for k in keys:
            if random.random() > 0.5:
                # This is the case that synthetic is Anchor, and real argument is positive
                second_random = random.random()
                anchor = query_dict[k]
                positive = arguana_corpus[k]
                if second_random > 0.75:
                    # Case that anchor breakdown, positive breakdown
                    anchor = np.random.choice(anchor['breakdown'])
                    positive = np.random.choice(positive['breakdown'])
                elif 0.5 < second_random <= 0.75:
                    # Case that anchor breakdown, positive not breakdown
                    anchor = np.random.choice(anchor['breakdown'])
                    positive = positive['query']
                elif 0.25 < second_random <= 0.5:
                    # Case that anchor not breakdown, positive breakdown
                    anchor = anchor['query']
                    positive = np.random.choice(positive['breakdown'])
                else:
                    # Case that anchor not breakdown, positive not breakdown
                    anchor = anchor['query']
                    positive = positive['query']
                curr_pair = (preprocessing(anchor), preprocessing(positive))
            else:
                # This is the case that synthetic is Anchor, and real argument is positive
                second_random = random.random()
                anchor = arguana_corpus[k]
                positive = query_dict[k]
                if second_random > 0.75:
                    # Case that anchor breakdown, positive breakdown
                    anchor = np.random.choice(anchor['breakdown'])
                    positive = np.random.choice(positive['breakdown'])
                elif 0.5 < second_random <= 0.75:
                    # Case that anchor breakdown, positive not breakdown
                    anchor = np.random.choice(anchor['breakdown'])
                    positive = positive['query']
                elif 0.25 < second_random <= 0.5:
                    # Case that anchor not breakdown, positive breakdown
                    anchor = anchor['query']
                    positive = np.random.choice(positive['breakdown'])
                else:
                    # Case that anchor not breakdown, positive not breakdown
                    anchor = anchor['query']
                    positive = positive['query']
                curr_pair = (preprocessing(anchor), preprocessing(positive))
            
            pairs.append(curr_pair)
    
        if shuffle:
            random.shuffle(pairs)
    
        train_trip_list = pairs
        test_trip_list = []
    
        print(f"Prepare data for ArguAna, train_length: {len(train_trip_list)}, test_length: {len(test_trip_list)}")
    
        with open(f"./data_prep/train_test_data/train_{experiment_name}.pickle", "wb") as f:
            pickle.dump(train_trip_list, f)

    else:

        query_dict = {}
        for q in query:
            query_dict[q['id']] = {'query': q['query']}
    
        keys = list(query_dict.keys())[:query_num]
    
        if 'arguana' not in experiment_name:
            raise ValueError("Experiment_name has no Arguana")
    
        pairs = []
    
        for k in keys:
            if random.random() > 0.5:
                curr_pair = (preprocessing(query_dict[k]['query']), preprocessing(arguana_corpus[k]['query']))
            else:
                curr_pair = (preprocessing(arguana_corpus[k]['query']), preprocessing(query_dict[k]['query']))
            
            pairs.append(curr_pair)
    
        if shuffle:
            random.shuffle(pairs)
    
        train_trip_list = pairs
        # test_trip_list = pairs[int(len(pairs)*0.9):]
        test_trip_list = []
    
        print(f"Prepare data for ArguAna, train_length: {len(train_trip_list)}, test_length: {len(test_trip_list)}")
    
        with open(f"./data_prep/train_test_data/train_{experiment_name}.pickle", "wb") as f:
            pickle.dump(train_trip_list, f)
    


def get_pair_data(experiment_name, query_num, query, dataset, d_fake, shuffle):
    query_dict = {}
    for q in query:
        query_dict[q['id']] = {'query': q['query']}

    keys = list(query_dict.keys())[:query_num]
    

    if experiment_name.split("_")[0] == "batch":
        print("preparing paired data with one anchor and positive, NO Negative")
        if "wtb" in experiment_name:
            print("Confirming the dataset is WTB")
            pairs = get_pair_large_batch_wtb(keys, query_dict, dataset)
        else:
            print("Confirming the dataset is DM")
            pairs = get_pair_large_batch(keys, query_dict, dataset)
    else:
        raise ValueError
    
    if shuffle:
        random.shuffle(pairs)

    train_trip_list = pairs
    
    train_trip = []
    for trip in train_trip_list:
        for each in trip:
            train_trip.append(each)

    if shuffle:
        random.shuffle(train_trip)
    print(len(train_trip))

    with open(f"./data_prep/train_test_data/train_{experiment_name}.pickle", "wb") as f:
        pickle.dump(train_trip, f)
