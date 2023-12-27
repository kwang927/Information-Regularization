import threading
import pickle
import argparse
import os
import math
import time
import openai
openai.api_key = None
import concurrent.futures

# This file contains function for GPT-generation, it should take in a dictionary of prompts, and output the generation result to the given path. This program is called in run_generation.sh

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def gpt_new_conversation(prompt, temperature=0, pp=0, max_tokens=3000):
    """
    Input: The prompt temperature (default 0) and presence penalty (default 0)
    Output: A list of dict that contains the user prompt and ChatGPT's answer to the prompt
    """
    try:
        conversation = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=conversation,
            temperature=temperature,
            request_timeout=90,
            presence_penalty=pp,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0)
        content = response.choices[0]['message']['content']
        # Remove the first sentence of the answer
        role = response.choices[0]['message']['role']
        conversation.append({'role': role, 'content': content})
        # This can be useful to calculate the cost
        # cost = response.usage['total_tokens']

        return conversation[-1]['content']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(5)
        return gpt_new_conversation(prompt, temperature)



def gen_query(prompt_dict, path_to_store):
    curr_id_list = list(prompt_dict.keys())
    curr_prompt_list = []
    for i in curr_id_list:
        curr_prompt_list.append(prompt_dict[i])

    generated_queries = {}

    for ind, prompt in enumerate(curr_prompt_list):
        curr_generated_query = gpt_new_conversation(prompt, max_tokens=3000)
        generated_queries[curr_id_list[ind]] = curr_generated_query

    create_directory_if_not_exists(f"{path_to_store}/")

    with open(f"{path_to_store}/query.pickle", 'wb') as f:
        pickle.dump(generated_queries, f)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_dir', required=True, help='Output file to write responses to')
    parser.add_argument('-t', '--thread_num', required=True, help='Number of threads used in the program')
    parser.add_argument('-s', '--index_to_start', required=True, help='The index to start')
    parser.add_argument('-e', '--index_to_end', required=True, help='The index to end')
    parser.add_argument('-prompt', '--prompt_file_to_read', required=True, help='The file that contains the prompts')

    args = parser.parse_args()

    file_name = args.prompt_file_to_read
    
    print(f"curr_filename: {file_name}")

    # print(f"Loading documents: {file_name}")

    with open(file_name, 'rb') as f:
        whole_prompt_dict = pickle.load(f)

    # print("Data loaded")

    output_path = args.output_dir
    thread_num = int(args.thread_num)
    start_ind = int(args.index_to_start)
    end_ind = int(args.index_to_end)
    num_of_queries = end_ind - start_ind
    path_head = output_path
    abs_id_list = []

    thread_length = math.ceil((num_of_queries / thread_num))
    thread_list = []
    thread_id = 0
    path_list = []
    for i in range(thread_num):

        start = int(thread_length * i)
        end = int(thread_length * (i + 1))



        curr_path = f"./generation_result/{path_head}/{start_ind + start}_{start_ind + end}" #edit folder name
        list_of_ids = list(whole_prompt_dict.keys())[start_ind + start: start_ind + end]

        prompt_dict = {}
        for i in list_of_ids:
            prompt_dict[i] = whole_prompt_dict[i]

        thread_list.append(
            threading.Thread(target=gen_query,
                             args=(prompt_dict, curr_path)))
        thread_id += 1
        path_list.append(curr_path)
        # time.sleep(5)
    # pdb.set_trace()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    all_queries = {}
    all_prompts = {}
    for p in path_list:
        with open(f"{p}/query.pickle", 'rb') as f:
            curr_queries = pickle.load(f)
        for k in curr_queries:
            all_queries[k] = curr_queries[k]
            
    with open(f"./generation_result/{path_head}/all_queries.pickle", 'wb') as f: # edit folder name
        pickle.dump(all_queries, f)



    print("Generation Ended")
    end_time = time.time()

