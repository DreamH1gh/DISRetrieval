import json
import re
from EmbeddingModel import SBertEmbeddingModel
from TreeStructure import Tree, Node
from TreeRetriever import TreeRetriever
from QAModel import UnifiedQAModel
import torch
import argparse
import os
from tqdm import tqdm

def dict_to_node(node_dict: dict) -> Node:
    """
    transform the discourse tree dictory into Node structure
    """
    left = dict_to_node(node_dict['left_child']) if not node_dict.get('is_leaf', False) and 'left_child' in node_dict else None
    right = dict_to_node(node_dict['right_child']) if not node_dict.get('is_leaf', False) and 'right_child' in node_dict else None

    return Node(
        node_id=node_dict['node_id'],
        node_text=node_dict['node_text'],
        is_leaf=node_dict['is_leaf'],
        interval=node_dict['node_interval'],
        depth=node_dict.get('depth', 0),
        left_child=left,
        right_child=right
    )

def build_tree(embedding_model, rst_file, device):
    """
    Build a Tree object from the given RST file:
    - Load the JSON-based tree structure.
    - Convert it to a Node-based structure.
    - Compute and update embeddings for all nodes.
    """
    with open(rst_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rst_tree = data['full_tree']
    root_node = dict_to_node(rst_tree)
    tree = Tree(root_node=root_node, device=device)

    node_texts = tree.get_node_text()
    node_ids = list(node_texts.keys())
    texts = [node_texts[nid] for nid in node_ids]
    with torch.no_grad():
        embeddings = embedding_model.create_embedding(texts).to(device)
    embeddings_map = {nid: emb for nid, emb in zip(node_ids, embeddings)}
    tree.update_embedding(embeddings_map)

    return tree

def retrieve_and_generate(config, device):
    """
    - Load or build trees for documents.
    - Perform tree-based retrieval based on a question.
    - Generate answers using a QA model.
    - Save results back to the question data.
    """
    rst_tree_path = os.path.join(config.data_path, 'doc_data', 'final_tree')
    tree_path = os.path.join(config.data_path, 'tree_data')
    question_path = os.path.join(config.data_path, 'question_map.json')

    embedding_model = SBertEmbeddingModel(device=device)
    qa_model = UnifiedQAModel(device=device)
    retriever = TreeRetriever(config=config, device=device)

    with open(question_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in tqdm(data, desc="Answering questions"):
        question = item['question']
        doc_id = item['doc_id']
        tree_file = os.path.join(tree_path, doc_id)

        if os.path.exists(tree_file):
            tree = Tree(device=device)
            tree.load(path=tree_file)
        else:
            rst_file = os.path.join(rst_tree_path, f'{doc_id}.json')
            tree = build_tree(embedding_model=embedding_model, rst_file=rst_file, device=device)
            tree.save(path=tree_file)
        
        retriever.tree = tree
        query_embedding = embedding_model.create_embedding([question])[0]
        evidence = retriever.retrieve(query_embedding=query_embedding)

        answer = qa_model.answer_question(context=evidence, question=question)

        item['predicted_evidence'] = evidence
        item['predicted_answer'] = answer
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/quality',
        help="root path of dataset")
    
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='out.json',
        help="save path for retrieval and QA results")
    
    parser.add_argument(
        '--max_node_depth', 
        type=int, 
        default=10000,
        help="max depth of retrieved node. set to 1 to only retrieve leaves, and set to a very large number to retrieve all nodes")
    
    parser.add_argument(
        '--rank_order', 
        action='store_true', 
        default=False,
        help='whether use the rank order for the top-k leaves of a retrieved intermediate node. default to False means use the original order in document')
    
    parser.add_argument(
        '--max_len', 
        type=int, 
        default=200,
        help='max number of retrieved context words')
    
    parser.add_argument(
        '--leaf_topk', 
        type=int, 
        default=5,
        help="value of k to filter the top-k leaves of retrieved intermediate node")
    
    ## base, sum, rest_topk
    parser.add_argument(
        '--method', 
        type=str, 
        default='rest_topk',
        help="different retrieval methods. base means use all leaves of a intermediate node. sum means use the summary of intermediate node. rest_topk is our method.")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    retrieve_and_generate(args, device)