# Information-Regularization
## Paper abstract
Effective information retrieval (IR) in settings with limited training data, particularly for complex queries, remains a challenging task. This paper introduces a method of Information Regularization for synthetic query generation aimed at improving data augmentation techniques and consequently, IR systems, by preventing models from learning superficial features of queries. Our approach, representing a novel application of regularization techniques in synthetic data creation for IR, is tested on three recent IR tasks characterized by complex queries: DORIS-MAE, ArguAna, and WhatsThatBook. Experimental results indicate that our regularization techniques not only outperform previous synthetic query generation methods on the tasks considered but also reduce cost by up to 50\%. Furthermore, this paper categorizes and explores three regularization methods at different stages of the query synthesis pipeline—input, prompt, and output—each offering varying degrees of performance improvement compared to models where no regularization is applied. This provides a systematic approach for optimizing synthetic data generation in data-limited, complex-query IR scenarios.

## Setup
We highly recommend creating a new conda environment for the following steps by:
```
conda create -n info_reg python=3.10
conda activate info_reg
```
To start, please run
```
bash setup.sh
```
Running the above line will create a new conda environment with name `info_reg`, download the needed packages and datasets, and unzip the needed files.

## Generation Process
Using the data from `evaluation/dataset/` and following prompts as shown in `prompt/`, the generated queries are stored in `generation/`.

## Generation

To generate results from gpt:

1. **Set up your API Key and OpenAI model**:
   - Open the `gpt_generation.py` file.
   - Locate the line `openai.api_key = None` and replace `None` with your OpenAI API key.
   - Our default model is `"gpt-4-0613"`, change this according to your requirement.

2. **Configure the Shell Script**:
   - Open the `run_generation.sh` script in a text editor.
   - Set the following variables according to your requirements:
     - `thread_num`: Number of threads, usually `20`
     - `prompt_path`: Path to the prompt file, this should be a pickle that stores a dictionary of format id string to prompt string.
     - `name`: prefix of name for the output file, e.g., `"arguana_Dreg_40%"`.
     - `start`: Starting index for processing, usually `0`.
     - `end`: Ending index for processing. This depends on the length of the prompt file

3. **Run the Generation Script**:
   - Execute the `run_generation.sh` script to start the generation process.
   - We recommend using nohup to run the generation process
     ```bash
     nohup bash run_generation.sh > generation_log/<LOG_FILE> 2>&1 &  
     ```
     remember to replace <LOG_FILE> with name of the log file


## Training & Evaluation
To train an embedding model using synthetically generated data, run the following
```bash
python3 -u run_training_evaluation.py -query_type <QUERY_TYPE> -num_experiment <NUM_EXPERIMENTS> -name <EXPERIMENT_NAME> -query_num <QUERY_NUMBER> -half <WHETHER_FREEZE_HALF> -shuffle <SHUFFLE> -cuda <CUDA> -margin <MARGIN> -batch_size <TRAINING_BATCHSIZE> -model_name <MODEL> -evaluation <WHETHER_RUN_EVALUATION>
```

- `<QUERY_TYPE>`: Type of query ('old' or 'new').
- `<NUM_EXPERIMENTS>`: Number of experiments to run.
- `<EXPERIMENT_NAME>`: Name for the experiment.
- `<QUERY_NUMBER>`: Number of queries to process.
- `<WHETHER_FREEZE_HALF>`: Set to `True` or `False` depending on whether to train half of the layer.
- `<SHUFFLE>`: Set to `True` or `False` to shuffle the data.
- `<CUDA>`: The CUDA used to train the model
- `<MARGIN>`: Margin of the contrastive training
- `<TRAINING_BATCHSIZE>`: Training batch size
- `<MODEL>`: The model to train/evaluate
- `<WHETHER_RUN_EVALUATION>`: Set to `True` or `False` to decide whether to run evaluation or not


Replace each `<PLACEHOLDER>` with the appropriate value for your experiment.

The result checkpoint will be stored at `training/model_checkpoints`
