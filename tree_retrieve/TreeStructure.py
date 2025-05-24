# tree.py

import torch
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field

@dataclass
class Node:
    node_id: str
    node_text: str
    is_leaf: bool
    interval: List[Any]
    depth: int = 0
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)


class Tree:
    def __init__(self, root_node: Node=None, device: torch.device=None):
        self.root = root_node
        self.device = device
        if self.root:
            self._add_node_depth()

    def _add_node_depth(self) -> None:
        """
        calculate the depth of each node. The depth of a node is defined as the max number of edges from the node to the subtree leaves. depth is set to 1 for leaf nodes.
        """
        def add_depth(node):
            if node.is_leaf:
                node.depth = 1
                return 1
            else:
                child_depth = max(add_depth(node.left_child), add_depth(node.right_child))
                node.depth = child_depth + 1
                return child_depth + 1
        add_depth(self.root)

    def update_embedding(self, embeddings_map: dict[str, torch.Tensor]) -> None:
        """
        update node embeddings
        """
        def update_node(node):
            node.embedding = embeddings_map[node.node_id]
            if not node.is_leaf:
                update_node(node.left_child)
                update_node(node.right_child)
        update_node(self.root)

    def get_node_text(self) -> Dict[str, str]:
        """
        get node text map of full tree

        Return:
            dict{node_id: node_text}
        """
        text_map = {}

        def node_text(node):
            id = node.node_id
            text_map[id] = node.node_text
            if not node.is_leaf:
                node_text(node.left_child)
                node_text(node.right_child)
        
        node_text(self.root)

        return text_map

    def get_all_nodes(self) -> List[Node]:
        """
        get all nodes of the full tree
        """
        nodes = []
        def gather(node):
            nodes.append(node)
            if not node.is_leaf:
                gather(node.left_child)
                gather(node.right_child)
        gather(self.root)
        return nodes

    def get_leaf_nodes(self, node: Node) -> List[Node]:
        """
        get all leaves on a subtree of an intermediate node
        """
        leaf_nodes = []
        def collect(n):
            if n.is_leaf:
                leaf_nodes.append(n)
            else:
                collect(n.left_child)
                collect(n.right_child)
        collect(node)
        return leaf_nodes

    def get_node_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        get embedding map of full tree
        Return:
            dict{node_id: node_embedding}
        """
        embedding_map = {}
        def gather(node):
            if node.embedding is not None:
                embedding_map[node.node_id] = node.embedding
            if not node.is_leaf:
                gather(node.left_child)
                gather(node.right_child)
        gather(self.root)
        return embedding_map

    def save(self, path: str) -> None:
        """
        save tree
        """
        torch.save(self.root, path)

    def load(self, path: str) -> None:
        """
        load tree
        """
        self.root = torch.load(path)
        self._add_node_depth()
