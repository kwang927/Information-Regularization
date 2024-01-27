import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW as AdamW
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import pdb


class MyDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def linear_warmup_scheduler(current_step, max_lr, warmup_steps, optimizer):
    lr = (current_step / warmup_steps) * (max_lr)

    for param in optimizer.param_groups:
        param['lr'] = lr

    return optimizer


def load_pretrained_model(model_name, half):

    if model_name == "ernie":
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-large-en")
        model = AutoModel.from_pretrained("nghuyong/ernie-2.0-large-en")
    elif model_name == "simlm":
        model = AutoModel.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
        tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
    elif model_name == "spladev2":
        model = AutoModel.from_pretrained('naver/splade_v2_distil')
        tokenizer = AutoTokenizer.from_pretrained('naver/splade_v2_distil')
    elif model_name == "scibert":
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')   
    elif model_name == "e5":
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        model = AutoModel.from_pretrained('intfloat/e5-large-v2')
    elif model_name == "specterv2":
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        model = AutoModel.from_pretrained('allenai/specter2_base')
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    elif model_name == "simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')


    if half:

        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        half_length = len(model.encoder.layer)//2
        for layer in model.encoder.layer[half_length:]:
            for param in layer.parameters():
                param.requires_grad = True

    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f" trainable parameters: {int(num_train_params / 1000000)} millions, total parameters {int(num_params / 1000000)} millions.")

    return tokenizer, model


def model_process(curr_batch, model, tokenizer, model_name):

    if model_name in ["e5", "simcse", "roberta", "specterv2", "simlm", "spladev2", "scibert", "ernie"]:
        batch_dict = tokenizer(curr_batch, padding='max_length', max_length=32, truncation=True, return_tensors='pt')

    if model_name == "e5":
        input_ids = batch_dict['input_ids'].to("cuda")
        token_type_ids = batch_dict['token_type_ids'].to("cuda")
        attention_mask = batch_dict['attention_mask'].to("cuda")

        new_batch_dict = {}
        new_batch_dict["input_ids"] = input_ids
        new_batch_dict["token_type_ids"] = token_type_ids
        new_batch_dict["attention_mask"] = attention_mask

        outputs = model(**new_batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    elif model_name in ["simcse", "ernie", "scibert"]:
        input_ids = batch_dict['input_ids'].to("cuda")
        attention_mask = batch_dict['attention_mask'].to("cuda")

        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.pooler_output

        embeddings = F.normalize(outputs, p=2, dim=1)
    
    elif model_name == "roberta":
        input_ids = batch_dict['input_ids'].to("cuda")
        attention_mask = batch_dict['attention_mask'].to("cuda")

        new_batch_dict = {}
        new_batch_dict["input_ids"] = input_ids
        new_batch_dict["attention_mask"] = attention_mask

        outputs = model(**new_batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    elif model_name in ["simlm", "specterv2"]:
        input_ids = batch_dict['input_ids'].to("cuda")
        attention_mask = batch_dict['attention_mask'].to("cuda")
        embeddings = model(input_ids, attention_mask = attention_mask)
        outputs = embeddings.last_hidden_state[:, 0, :]
        embeddings = F.normalize(outputs, p=2, dim=1)

    elif model_name == "spladev2":
        input_ids = batch_dict['input_ids'].to("cuda")
        attention_mask = batch_dict['attention_mask'].to("cuda")
        embeddings = model(input_ids, attention_mask = attention_mask)
        outputs = (torch.sum(embeddings.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) /\
                                                  torch.sum(attention_mask, dim=-1, keepdim=True))
        embeddings = F.normalize(outputs, p=2, dim=1)
            
    else:
        raise ValueError

    return embeddings 
        
    
# Integrated code:
def training_model_large_batch_contrastive(batch_size, num_epochs, name, cuda, margin, shuffle, half, model_name, max_breach =5):
    model_path = "model_checkpoints"

    tokenizer, model = load_pretrained_model(model_name, half)
    
    torch.cuda.set_device(int(cuda.split(",")[0]))
    model.to("cuda")
    cuda_list = cuda.split(",")
    cuda_num = len(cuda_list)
    if cuda_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(idx) for idx in cuda_list])

    with open(f"data_prep/train_test_data/train_gemini_{name}.pickle", "rb") as f:
        train_data = pickle.load(f)

    print(f"length of train data triplets {len(train_data)}")
    
    assert batch_size >1  
    
    train_dataset = MyDataset(train_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, drop_last = True)  
    
    lr = 1e-5

    optimizer = AdamW(model.parameters(), lr=lr)
    
    accumulation = batch_size 

    T_0 = len(train_dataset) // accumulation

    print(f"T0 for scheduler is {T_0}")

    scheduler = Scheduler(optimizer=optimizer,
                          T_0=T_0)  

    delay = accumulation // batch_size 
    
    print(f"delay is {delay}, batch_size is {batch_size}")

    tau = 0.05


    print("training begin....")

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')

        model.train()
        losses = []

        loss = torch.tensor(0.0).to("cuda")
        for step, batch in enumerate(train_dataloader):
            
            assert batch_size == len(batch[0])
            assert len(batch) == 2
            assert len(batch[0]) == len(batch[1])
            assert batch[0]==batch[1]
            curr_batch = []
            curr_batch += list(batch[0])
            curr_batch += list(batch[1])

            embeddings = model_process(curr_batch, model, tokenizer, model_name)

            query_embedding = embeddings[:len(batch[0])]
            
            document_embedding = embeddings[len(batch[0]):]

            qd_sim = F.cosine_similarity(query_embedding.unsqueeze(1),document_embedding.unsqueeze(0), dim= 2)

            assert qd_sim.shape[0]==batch_size
            assert qd_sim.shape[1] == batch_size

            temp_loss = torch.tensor(0.0).to("cuda")
            for ind in range(batch_size):
                temp_loss -= torch.log(torch.exp(qd_sim[ind,ind]/tau)/torch.sum(torch.exp(qd_sim[ind,:]/tau)))
            temp_loss /= batch_size

            loss += temp_loss.detach()
            temp_loss.backward()  # compute gradients

            if (step + 1) % delay == 0:
                optimizer.step()  # update parameters
                scheduler.step()
                optimizer.zero_grad()  # reset gradients

                losses.append((loss / delay).item())
                print(f"finished {100*(step/len(train_dataloader))}%,  train loss this minibatch {losses[-1]}")
                
                loss = torch.tensor(0.0).to("cuda")


        print(f"Loss after epoch {epoch + 1}: {np.mean(losses)}")

        torch.save(model, f'./training/{model_path}/{name}_model_{model_name}_{epoch + 1}.pth')
        if model_name == "specterv2":
            model.module.save_adapter(f"./training/{model_path}/{name}_model_{model_name}_{epoch + 1}_adapter", "specter2")
        print("finish saving epoch stats")

    print('Training finished.')
