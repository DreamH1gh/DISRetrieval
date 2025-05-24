# retriever.py

import torch
import torch.nn.functional as F
import re
from typing import List, Any
from TreeStructure import Tree


def get_text(evidence: List[str]) -> str:
    text = ""
    for t in evidence:
        text += f"{' '.join(t.splitlines())}".strip() + "\n\n"
    return text


class TreeRetriever:
    def __init__(self, tree: Tree = None, config: Any = None, device: torch.device = None):
        self.tree = tree
        self.config = config
        self.device = device

    def calculate_score(self, query_embedding: torch.Tensor) -> dict[str, float]:
        """
        calculate scores of between the query and all node text
        """
        node_embeddings = self.tree.get_node_embeddings()
        if not node_embeddings:
            return {}

        node_ids = list(node_embeddings.keys())
        embedding_tensor = torch.stack([node_embeddings[nid] for nid in node_ids]).to(self.device)
        query_embedding = query_embedding.to(self.device).unsqueeze(0)

        scores_tensor = F.cosine_similarity(query_embedding, embedding_tensor)
        return {node_id: score.item() for node_id, score in zip(node_ids, scores_tensor)}

    def calculate_score_reranker(self, query: str) -> dict[str, float]:
        node_texts = self.tree.get_node_text()
        keys, pairs = [], []
        for k, v in node_texts.items():
            keys.append(k)
            pairs.append([query, v])

    def get_target_list(self, query_embedding: torch.Tensor):
        """
        get the full node list and cosine scores
        """
        scores = self.calculate_score(query_embedding)
        nodes = self.tree.get_all_nodes()

        filtered = [
            (node, scores[node.node_id])
            for node in nodes
            if node.node_id in scores and node.depth <= self.config.max_node_depth
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        if filtered:
            return zip(*filtered)
        return [], []

    def truncate_text(self, text: str, max_words: int) -> str:
        """
        cut off the retrieved context
        """
        matches = list(re.finditer(r'\S+', text))
        if len(matches) <= max_words:
            return text
        return text[:matches[max_words - 1].end()]

    def base_retrieve(self, query_embedding: torch.Tensor) -> str:
        """
        ablation method
        directly replace the intermediate nodes with all their subtree leaves
        """
        nodes, scores = self.get_target_list(query_embedding)
        texts, ids = [], set()

        for node in nodes:
            if node.is_leaf:
                if node.node_id not in ids:
                    texts.append(node.node_text)
                    ids.add(node.node_id)
            else:
                for leaf in self.tree.get_leaf_nodes(node):
                    if leaf.node_id not in ids:
                        texts.append(leaf.node_text)
                        ids.add(leaf.node_id)
            context = ' '.join(texts)
            if len(list(re.finditer(r'\S+', context))) > self.config.max_len:
                break

        return self.truncate_text(get_text(texts), self.config.max_len)

    def sum_retrieve(self, query_embedding: torch.Tensor) -> str:
        """
        ablation method
        directly use the summary of intermediate nodes
        """
        nodes, scores = self.get_target_list(query_embedding)
        texts, ids = [], set()

        for node in nodes:
            if node.node_id not in ids:
                texts.append(node.node_text)
                ids.add(node.node_id)

            context = ' '.join(texts)
            if len(list(re.finditer(r'\S+', context))) > self.config.max_len:
                break

        return self.truncate_text(get_text(texts), self.config.max_len)

    def rest_topk_retrieve(self, query_embedding: torch.Tensor) -> str:
        """
        our method. 
        find all subtree leaves of the intermediate node.
        filter leaves ranking before the intermediate node
        choose top-k of the rest node
        reorder into the sequence in original document
        replace the intermediate node with chosen leaves
        """
        nodes, scores = self.get_target_list(query_embedding)
        scores_dict = {node.node_id: score for node, score in zip(nodes, scores)}
        texts, ids = [], set()

        for node in nodes:
            if node.is_leaf:
                if node.node_id not in ids:
                    texts.append(node.node_text)
                    ids.add(node.node_id)
            else:
                leaf_nodes = self.tree.get_leaf_nodes(node)
                rest_nodes = [n for n in leaf_nodes if n.node_id not in ids]

                rest_nodes.sort(key=lambda n: scores_dict.get(n.node_id, 0), reverse=True)
                rest_nodes = rest_nodes[:self.config.leaf_topk]

                if not self.config.rank_order:
                    rest_nodes.sort(key=lambda n: n.interval[0])

                for n in rest_nodes:
                    texts.append(n.node_text)
                    ids.add(n.node_id)

            context = ' '.join(texts)
            if len(list(re.finditer(r'\S+', context))) > self.config.max_len:
                break

        return self.truncate_text(get_text(texts), self.config.max_len)

    def retrieve(self, query_embedding: torch.Tensor, method: str = "rest_topk") -> str:
        """
        choose different retrieve method
        """
        method_map = {
            "base": self.base_retrieve,
            "sum": self.sum_retrieve,
            "rest_topk": self.rest_topk_retrieve,
        }

        if method not in method_map:
            raise ValueError(f"Unknown retrieval method: {method}")

        return method_map[method](query_embedding)