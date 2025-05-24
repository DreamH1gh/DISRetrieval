import json
import os
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from utils import read_sent, parse_bracket_tree
from typing import List, Dict, Any

class RSTProcessor:

    def process_rst_and_construct_trees(self, rst_file: str, paragraph_edu_file: str, tree_path: str, sent_map_path: str) -> None:
        """
        read rst parser output and construct tree

        Args:
            rst_file: rst output file path
            paragraph_file: paragraph file 
            tree_path: tree save path
            sent_map: map from the word splited sentence to the original sentence
        Return:
            None

        """
        with open(paragraph_edu_file, 'r') as f:
            edu_data = json.load(f)

        with open(sent_map_path, 'r') as f:
            sent_map = json.load(f)

        with open(rst_file, 'r') as f:
            for sentences in read_sent(f):
                assert '# newdoc id =' in sentences[-2], f'{sentences[-2]}'
                doc_id = sentences[-2].split('=')[-1].strip()
                tree = sentences[-1]

                if '{[(error)]}' in tree:
                    continue

                temp = edu_data[doc_id]
                w_data = {'id': doc_id, **temp}
                w_data['tree'] = tree

                tokens = []
                for sent in temp['sentences']:
                    sent_tokens = sent.split(' ')
                    for tok in sent_tokens:
                        token_text = '_'.join(tok.split('_')[:-1])
                        tokens.append(token_text)

                assert len(tokens) == temp['edus'][-1][1] + 1, f"Token count mismatch for {doc_id}"

                w_data['tree_str'] = tree
                tree_node = parse_bracket_tree(tree)
                w_data['full_tree'] = self._update_node(node=tree_node, tokens=tokens, intervals=temp['edus'], id='root', sent_map=sent_map[doc_id])


                with open(f'{tree_path}/{doc_id}.json', 'w') as out_f:
                    json.dump(w_data, out_f, ensure_ascii=False, indent=4)


    def create_recursive_paragraphs(self, subtree_path: str, new_paragraph_edu_file: str) -> None:
        """
        Merge root noods of paragraph-level subtrees, create high level sub-document for the second stage of recursive discourse parsing

        Args:
            subtree_path: path to the subtrees
            new_paragraph_edu_file: path to save the created sub-document

        Return:
            None
        """
        file_map = {}
        for filename in os.listdir(subtree_path):
            filename = filename.replace('.json', '').split('_')
            doc_id = '_'.join(filename[:-1])
            idx = filename[-1]
            file_map.setdefault(doc_id, []).append(int(idx))

        merged = {}
        sentence_map = {}
        for doc_id, indices in tqdm(file_map.items(), desc="Merging trees"):
            sentences = []
            edus = []
            start = 0
            sentence_map[doc_id] = {}
            for idx in sorted(indices):
                fp = os.path.join(subtree_path, f'{doc_id}_{idx}.json')
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                text = data['full_tree']['node_text']
                try:
                    tokens = word_tokenize(text)
                except:
                    print(fp)
                    break

                sentence_map[doc_id][' '.join(tokens)] = text
                sentence_map[doc_id][text] = text

                tokens_tagged = [t + '_NNP' for t in tokens]
                tokens_tagged.append('<S>')
                sentences.append(' '.join(tokens_tagged))
                edus.append((start, start + len(tokens_tagged) - 1, '<S>'))
                start += len(tokens_tagged)

            merged[doc_id] = {'sentences': sentences, 'edus': edus}

        with open(new_paragraph_edu_file, 'w') as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        
        with open(new_paragraph_edu_file.replace('.json', '_sentmap.json'), 'w') as f:
            json.dump(sentence_map, f, ensure_ascii=False, indent=4)


    def merge_tree(self, highlevel_tree_path: str, subtree_path: str, final_tree_path: str) -> None:
        """
        replace leaf nodes of high level tree with corresponding paragraph-level subtree

        Args:
            highlevel_tree_path: path of high level trees constructed from root nodes of paragraph-level subtrees
            subtree_path: paragraph-level subtree path
            final_tree_path: final discourse tree path

        Return: 
            None
        """
        for filename in tqdm(os.listdir(highlevel_tree_path), desc="Merging document trees"):
            doc_id = filename.replace('.json', '')
            with open(os.path.join(highlevel_tree_path, filename), 'r') as f:
                doc_data = json.load(f)

            total_tree = doc_data['full_tree']
            leaf_ids = []
            self._collect_leaf_ids(total_tree, leaf_ids)
            leaf_map = {leaf_id: idx for idx, leaf_id in enumerate(leaf_ids)}

            def _merge(node):
                if node['is_leaf'] and node['node_id'] == 'root':
                    doc_data['full_tree'] = self._replace_with_subtree(node, doc_id, 0, subtree_path)
                    return
                if node['is_leaf']:
                    assert node['node_id'] not in leaf_map, f"{doc_id}, {node['node_id']}"
                    return
                if node['left_child']['node_id'] in leaf_map:
                    node['left_child'] = self._replace_with_subtree(
                        node['left_child'], doc_id, leaf_map[node['left_child']['node_id']], subtree_path)
                else:
                    _merge(node['left_child'])

                if node['right_child']['node_id'] in leaf_map:
                    node['right_child'] = self._replace_with_subtree(
                        node['right_child'], doc_id, leaf_map[node['right_child']['node_id']], subtree_path)
                else:
                    _merge(node['right_child'])

            _merge(total_tree)
            self._update_all_node_intervals(node=total_tree, global_index=[0])

            with open(os.path.join(final_tree_path, f'{doc_id}.json'), 'w') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=4)

    # ==== Private helpers ====

    def _update_node(self, node: Dict, tokens: List[str], intervals: List[Any], id: str, sent_map: Dict) -> None:
        """
        add node information for newly constructed tree

        Args:
            node: tree node
            tokens: document token list
            intervals: stntence spans
            id: node id
            sent_map: map from the word splited sentence to the original sentence
        
        Return:
            None

        """
        label = node['label'].split(' ')
        node['node_interval'] = None
        node['node_id'] = id
        if len(label) == 4:
            node['is_leaf'] = True
            start  = int(label[-2])
            end = int(label[-1])
            node['left_child'] = None
            node['right_child'] = None
            split_text = ' '.join(tokens[start:end])
            node['node_text'] = sent_map[split_text]
            node.pop('children')
            for interval in intervals:
                if interval[0] == start and interval[1] == end:
                    node['node_interval'] = interval
                    break
            assert node['node_interval'] is not None
        else:
            node['is_leaf'] = False
            node['left_child'] = self._update_node(node=node['children'][0], tokens=tokens, intervals=intervals, id=id + '.l', sent_map=sent_map)
            node['right_child'] = self._update_node(node=node['children'][1], tokens=tokens, intervals=intervals, id=id + '.r', sent_map=sent_map)
            node.pop('children')
            node['node_text'] = None
            node['node_interval'] = [node['left_child']['node_interval'][0], node['right_child']['node_interval'][1]]

        return node

    def _update_node_id(self, node: Dict, new_root_id: str) -> None:
        """
        recursively update node ids of paragraph-level subtrees after merged
        """
        node['node_id'] = node['node_id'].replace('root', new_root_id)
        if not node['is_leaf']:
            self._update_node_id(node['left_child'], new_root_id)
            self._update_node_id(node['right_child'], new_root_id)

    def _collect_leaf_ids(self, node: Dict, leaf_list: List[str]) -> None:
        """
        collect all leaf node ids of high level tree for merging 
        """
        if node['is_leaf']:
            leaf_list.append(node['node_id'])
        else:
            self._collect_leaf_ids(node['left_child'], leaf_list)
            self._collect_leaf_ids(node['right_child'], leaf_list)

    def _replace_with_subtree(self, node: Dict, doc_id: str, index: int, subtree_path: str) -> dict:
        """
        reform subtree and return root node of subtree for merging
        """
        fp = os.path.join(subtree_path, f'{doc_id}_{index}.json')
        with open(fp, 'r') as f:
            sub_tree_data = json.load(f)
        sub_tree = sub_tree_data['full_tree']
        self._update_node_id(sub_tree, node['node_id'])
        return sub_tree

    def _update_all_node_intervals(self, node: Dict, global_index: List[int]) -> List[Any]:
        """
        recursively update node_interval of full tree
        """
        if node['is_leaf']:
            length = node['node_interval'][1] - node['node_interval'][0] + 1
            new_interval = (global_index[0], global_index[0] + length - 1)
            node['node_interval'] = new_interval
            global_index[0] += length
            return new_interval
        
        left_interval = self._update_all_node_intervals(node['left_child'], global_index)
        right_interval = self._update_all_node_intervals(node['right_child'], global_index)

        node['node_interval'] = (left_interval[0], right_interval[1])
        return node['node_interval']

