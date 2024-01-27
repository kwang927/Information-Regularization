conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
pip install beir
pip install tokenizers==0.12.1
pip install -U adapters

pip install transformers
pip install openai==0.27.8
pip install huggingface_hub==0.20.1
pip install --upgrade huggingface_hub
pip install nltk


# download dataset and unzip necessary files

mkdir ./training/model_checkpoints
mkdir ./data_prep/train_test_data

cd evaluation/dataset/
wget https://zenodo.org/record/8299749/files/DORIS-MAE_dataset_v1.json
mv DORIS-MAE_dataset_v1.json DORIS_MAE_dataset_v1.json
cd -

cd evaluation/
unzip arguana_result_template.pickle.zip 
cd -
