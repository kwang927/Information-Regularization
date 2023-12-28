import pytrec_eval
import logging
import pickle
import numpy as np
import pathlib, os
from typing import List, Dict, Tuple
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.custom_metrics import mrr
import logging
from get_ranking import get_embed_text

logger = logging.getLogger(__name__)

class OwnModel:
    def __init__(self, model=None, model_path=None, dataset_name=None, cuda='cpu', batch_size=50, **kwargs):
        self.model = model 
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.cuda = cuda
        self.batch_size = batch_size
        
    def cosine_similarity_distance(self, array1, array2):
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)

        if norm1 == 0 or norm2 == 0:
            return 0  # Handle division by zero

        cosine_distance = dot_product / (norm1 * norm2)
        return cosine_distance
    
    def L2_distance(self, array1, array2):
        return - np.linalg.norm(array1 - array2)
    
    def get_dataset(self):
        if self.dataset_name == "scifact":
            
            dataset = "scifact"
        elif self.dataset_name == "arguana":
            dataset = "arguana"
        elif self.dataset_name == "treccovid":
            with open("./needed_treccovid.pickle", 'rb') as f:
                dataset = pickle.load(f)
            return dataset

        print(f"Loading: {dataset}")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path("./beir_download/").absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        if self.dataset_name == "arguana":
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
        dataset = {'query': queries, 'corpus': corpus, 'qrels': qrels}
        return dataset
    
    def get_embedding(self):
        dataset = self.get_dataset()

        get_embed_text(dataset, self.dataset_name, self.model, self.batch_size, self.cuda, self.model_path)
        return
    
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:

        if self.model_path == None:
            file_path = f"./evaluation/embedding_result/{self.dataset_name}_embedding_result_of_{self.model}_pretrain.pickle"
        else:
            file_path = f"./evaluation/embedding_result/{self.dataset_name}_embedding_result_of_{self.model}_{self.model_path}.pickle"
        if os.path.exists(file_path) and 'pretrain' in file_path:
            with open(file_path, 'rb') as f:
                emb = pickle.load(f)
        else:
            self.get_embedding()
            with open(file_path, 'rb') as f:
                emb = pickle.load(f)


        text2id = emb['text2id_dict']
        emb = emb['embedding_result']

        print("Loaded files")



        scores = []
        for pair in sentences:
            key_to_look_up_1 = text2id[pair[0]]
            key_to_look_up_2 = text2id[pair[1]]

            q_emb = emb[key_to_look_up_1]
            c_emb = emb[key_to_look_up_2]

            if self.model in ['specterv2']:
                score = self.L2_distance(q_emb, c_emb)
            else:
                score = self.cosine_similarity_distance(q_emb, c_emb)
            
            scores.append(score)
        return scores
    
    def evaluate(self, qrels: Dict[str, Dict[str, int]], 
             results: Dict[str, Dict[str, float]], 
             k_values: List[int],
             ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        if ignore_identical_ids:
            logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        MRR_result = mrr(qrels, results, k_values)

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision, MRR_result
