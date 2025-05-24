import os
from SummerizationModel import LLamaAPISummerizationModel
from RSTProcesser import RSTProcessor
from utils import ensure_path_exits
import fire

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(BASE_DIR, ".."))
])
from rst_parser.driver.RSTParser import RSTParser

class TreeBuilder:
    def __init__(self, data_path):
        self.rst_parser = RSTParser()
        self.rst_processer = RSTProcessor()
        self.data_path = data_path
        self.paths = self._init_paths()
    
    def _init_paths(self):
        doc_data = os.path.join(self.data_path, 'doc_data')

        return {
            "subtree_edu": os.path.join(self.data_path, 'doc_data', 'subtree.json'),
            'sentence_map': os.path.join(self.data_path, 'doc_data', 'subtree_sentmap.json'),
            "highlevel_tree_edu": os.path.join(self.data_path, 'doc_data', 'highlevel_tree.json'),
            "subtree": os.path.join(doc_data, 'subtree'),
            "highlevel_tree": os.path.join(doc_data, 'highlevel_tree'),
            "final_tree": os.path.join(doc_data, 'final_tree')
        }

    def first_stage_parsing(self):
        """
        paragraph-level parsing and subtree construction
        """
        rst_file = self.rst_parser.start_parsing(edu_file=self.paths['subtree_edu'])
        ensure_path_exits(self.paths['subtree'])
        self.rst_processer.process_rst_and_construct_trees(rst_file=rst_file, 
                                                           paragraph_edu_file=self.paths['subtree_edu'],
                                                           tree_path=self.paths['subtree'],
                                                           sent_map_path=self.paths['subtree_edu'].replace('.json', '_sentmap.json'))

    def second_stage_parsing(self):
        """
        highlevel parsing using subtree roots
        """
        self.rst_processer.create_recursive_paragraphs(subtree_path=self.paths['subtree'],
                                                       new_paragraph_edu_file=self.paths['highlevel_tree_edu'])
        rst_file = self.rst_parser.start_parsing(edu_file=self.paths['highlevel_tree_edu'])
        ensure_path_exits(self.paths['highlevel_tree'])
        self.rst_processer.process_rst_and_construct_trees(rst_file=rst_file,
                                                           paragraph_edu_file=self.paths['highlevel_tree_edu'],
                                                           tree_path=self.paths['highlevel_tree'],
                                                           sent_map_path=self.paths['highlevel_tree_edu'].replace('.json', '_sentmap.json'))

    def final_stage(self):
        """
        merge two stage results into the full document-level tree
        """
        ensure_path_exits(self.paths['final_tree'])
        self.rst_processer.merge_tree(highlevel_tree_path=self.paths['highlevel_tree'],
                                      subtree_path=self.paths['subtree'],
                                      final_tree_path=self.paths['final_tree'])

def main(data_path, stage):
    builder = TreeBuilder(data_path)
    if stage == 'first':
        builder.first_stage_parsing()
    elif stage == 'second':
        builder.second_stage_parsing()
    elif stage == 'final':
        builder.final_stage()
    else:
        print("Invalid stage specified. Available stages: first, second, final.")

if __name__ == "__main__":
    fire.Fire(main)