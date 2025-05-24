import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import gc
import sys

def init_vllm(
    adapter_name_or_path: str = None,
    model_path: str = "~/hfmodel/Meta-Llama-3___1-8B-Instruct", 
    max_new_tokens: int = 1024,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    gpu_num: int = 1,
    do_sample: bool = True
):
    """
    Initialize the vLLM model, tokenizer, and sampling parameters.
    """
    if not do_sample:
        temperature = 0
        top_k = 1
        top_p = 1
        
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[],
        top_p=top_p,
        top_k=top_k
    )

    llm = LLM(
        model=model_path, 
        enable_lora=adapter_name_or_path is not None,
        trust_remote_code=True,
        tensor_parallel_size=gpu_num,
        # gpu_memory_utilization=0.7
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer, sampling_params

def get_text(node):
    """
    Recursively concatenate text from all leaf nodes of a subtree.
    """
    if node['is_leaf']:
        return node['node_text']
    else:
        return get_text(node['left_child']) + ' ' + get_text(node['right_child'])


def get_done_map(args):
    """
    Preprocess all trees and collect nodes whose text length is below a threshold.
    Save these as the initial summaries in a dictionary for further processing.
    """
    files = os.listdir(args.tree_path)
    done_map = {}
    if os.path.exists(args.save_path):
        return 
    for file in tqdm(files, desc="Processing trees"):
        with open(os.path.join(args.tree_path, file), 'r') as f:
            data = json.load(f)
            tree = data['full_tree']
            id = data['id']
            temp_map = {}

            def add_node(node):
                if node['is_leaf']:
                    temp_map[node['node_id']] = {'text':node['node_text']}
                else:
                    node_text = get_text(node)
                    if len(node_text.split()) < args.tau:
                        temp_map[node['node_id']] = {'text':node_text}
                    add_node(node['left_child'])
                    add_node(node['right_child'])
            
            add_node(tree)
            done_map[id] = temp_map
    
    with open(args.save_path, 'w') as f:
        json.dump(done_map, f, ensure_ascii=False, indent=4)


def get_tree_summary(node, tree_done_map, data_list):
    """
    Traverse the tree and collect pairs of child node summaries
    for which both children already have summaries.
    """

    lid = ''
    rid = ''
    if not node['is_leaf']:
        lid = node['left_child']['node_id']
        rid = node['right_child']['node_id']
    if node['node_id'] in tree_done_map.keys():
        return 
    elif lid in tree_done_map.keys() and rid in tree_done_map.keys():
        l_text = tree_done_map[lid]['text']
        r_text = tree_done_map[rid]['text']
        label = node['label'].split(' ')[0].strip()
        data_list.append({'node_id':node['node_id'], 'l':l_text, 'r':r_text, 'label':label})
        return 
    else:
        get_tree_summary(node['left_child'], tree_done_map, data_list)
        get_tree_summary(node['right_child'], tree_done_map, data_list)


def load_data(args, done_data):
    """
    Load all trees and prepare input pairs of summaries
    for which a new summary needs to be generated.
    """
    file_list = os.listdir(args.tree_path)
    summary_data = []
    
    for file in tqdm(file_list, desc="Preparing summary data"):
        with open(os.path.join(args.tree_path, file)) as f:
            tree_data = json.load(f)
            id = tree_data['id']
            data_list = []
            tree = tree_data['full_tree']
            get_tree_summary(tree, done_data[id], data_list)

            for item in data_list:
                item['tree_id'] = id
            summary_data.extend(data_list)

    return summary_data

def continue_summary(llm, tokenizer, sampling_params, args):
    """
    Use the language model to generate summaries for intermediate nodes
    based on summaries of their child nodes, and update the saved summary map.
    """

    prompt = 'write a summary of the given sentences, keeps as more key information as possible. Only give the summary without other text. Makse sure that the summary no more than 200 words.\ngiven text: \n'
    with open(args.save_path, 'r') as f:
        done_data = json.load(f)

    while True:
        sum_data = load_data(args, done_data)
        if len(sum_data) == 0:
            break

        inputs = []
        for item in sum_data:
            input = tokenizer.apply_chat_template([{'role':'user', 'content':f"{prompt} {item['l']} {item['r']}"}], add_generation_prompt=True, tokenize=False)
            inputs.append(input)

        results = llm.generate(inputs, sampling_params)
        for i, item in enumerate(sum_data):
            summary = results[i].outputs[0].text
            tree_id = item['tree_id']
            ndoe_id = item['node_id']
            done_data[tree_id][ndoe_id] = {'text':summary}

        with open(args.save_path, 'w') as f:
            json.dump(done_data, f, ensure_ascii=False, indent=4) 

def merge_result(args):
    """
    Merge generated summaries back into the original tree structure
    """

    files = os.listdir(args.tree_path)
    with open(args.save_path, 'r', encoding='utf-8') as f:
        mid_map = json.load(f)
    
    for file in tqdm(files, desc="Merging trees"):
        fp = os.path.join(args.tree_path, file)
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tree = data['full_tree']
        doc_id = data['id']
        
        def add_text(node):
            if not node['is_leaf']:
                id = node['node_id']
                node['node_text'] = mid_map[doc_id][id]['text']

                add_text(node['left_child'])
                add_text(node['right_child'])
        
        add_text(tree)

        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tree_path', 
        type=str, 
        default='data/quality/doc_data/highlevel_tree',
        help="path of trees to be summrized")
    
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='')
    
    parser.add_argument(
        '--gpu_num', 
        type=int, 
        default=2,
        help="used gpu number")
    
    parser.add_argument(
        '--gpu_id', 
        type=str, 
        default='0,1')
    
    parser.add_argument(
        '--tau', 
        type=int, 
        default=100,
        help="value of hyperparameter tau")
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='llama',
        help="used LLM for summarize")
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    from vllm import LLM, SamplingParams

    get_done_map(args)
    path_map = {
        'qwen':'~/hfmodel/Qwen2.5-7B-Instruct',
        'mistral':'~/GraphRag/qasper/vllm/Mistral-7B-Instruct-v0.2',
        'llama':'~/hfmodel/Meta-Llama-3___1-8B-Instruct'
    }
    llm, tokenizer, sampling_params = init_vllm(gpu_num=args.gpu_num, 
                                                model_path=path_map[args.model])

    try:
        continue_summary(llm, tokenizer, sampling_params, args)
        merge_result(args)
    finally:
        del llm
        del tokenizer
        gc.collect()
        sys.exit(0)