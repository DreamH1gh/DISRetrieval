import os
import sys
# sys.path.extend(["../../","../","./", "RSTparser"])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.extend([
    os.path.abspath(os.path.join(BASE_DIR, "..", "..")),
    os.path.abspath(os.path.join(BASE_DIR, ".."))
])

from data.Config import *
import pickle
from data.Dataloader import *
from modules.Parser import *
from modules.EDULSTM import *
from modules.Decoder import *
from modules.XLNetTune import *
from modules.TypeEmb import *
from data.TokenHelper import *
from modules.GlobalEncoder import *
from modules.Optimizer import *

import time
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
import torch

class RSTParser:
    def __init__(self, config_file='rst_parser/saved_model/100/config.cfg', use_cuda=True):
        self.config = Configurable(config_file)
        self.vocab = pickle.load(open(self.config.load_vocab_path, 'rb'))
        self.model_state_dict = torch.load(self.config.load_model_path)
        self.token_helper = TokenHelper(self.config.xlnet_save_dir)
        self.auto_extractor = AutoModelExtractor(self.config.xlnet_save_dir, self.config, self.token_helper)

        self.global_encoder = GlobalEncoder(self.vocab, self.config, self.auto_extractor)
        self.typeEmb = TypeEmb(self.vocab, self.config)
        self.EDULSTM_model = EDULSTM(self.vocab, self.config)
        self.decoder = Decoder(self.vocab, self.config)

        self.global_encoder.mlp_words.load_state_dict(self.model_state_dict["mlp_words"])
        self.global_encoder.rescale.load_state_dict(self.model_state_dict["rescale"])
        self.EDULSTM_model.load_state_dict(self.model_state_dict["EDULSTM"])
        self.typeEmb.load_state_dict(self.model_state_dict["typeEmb"])
        self.decoder.load_state_dict(self.model_state_dict["dec"])

        if use_cuda and torch.cuda.is_available():
            self.config.use_cuda = True
            self.global_encoder = self.global_encoder.cuda()
            self.typeEmb = self.typeEmb.cuda()
            self.EDULSTM_model = self.EDULSTM_model.cuda()
            self.decoder = self.decoder.cuda()
        else:
            self.config.use_cuda = False

        self.parser = DisParser(self.global_encoder, self.EDULSTM_model, self.typeEmb, self.decoder, self.config)

    def predict_one(self, doc):
        """predict one document"""
        self.parser.eval()
        onebatch = [(doc,)]

        try:
            doc_inputs = batch_doc_variable(onebatch, self.vocab, self.config, self.token_helper)
            EDU_offset_index, batch_denominator, edu_lengths, edu_types = batch_doc2edu_variable(
                onebatch, self.vocab, self.config, self.token_helper
            )

            with torch.no_grad():
                with autocast():
                    self.parser.encode(doc_inputs, EDU_offset_index, batch_denominator, edu_lengths, edu_types)
                    self.parser.decode(onebatch, None, None, self.vocab)

            cur_states = self.parser.batch_states[0]
            cur_step = self.parser.step[0]
            tree_str = cur_states[cur_step - 1]._stack[cur_states[cur_step - 1]._stack_size - 1].str
            return tree_str

        except Exception as e:
            print(f"[predict_one] Error: {e}")
            return '{[(error)]}'

    def predict_batch(self, data, output_file):
        """predict batch"""
        start = time.time()
        self.parser.eval()
        with open(output_file, mode='w', encoding='utf8') as outf:
            for onebatch in tqdm(data_iter(data, self.config.test_batch_size, False)):
                try:
                    doc_inputs = batch_doc_variable(onebatch, self.vocab, self.config, self.token_helper)
                    EDU_offset_index, batch_denominator, edu_lengths, edu_types = batch_doc2edu_variable(
                        onebatch, self.vocab, self.config, self.token_helper
                    )
                    with torch.no_grad():
                        with autocast():
                            self.parser.encode(
                                doc_inputs, EDU_offset_index, batch_denominator, edu_lengths, edu_types
                            )
                            self.parser.decode(onebatch, None, None, self.vocab)

                    for idx in range(len(onebatch)):
                        doc = onebatch[idx][0]
                        cur_states = self.parser.batch_states[idx]
                        cur_step = self.parser.step[idx]
                        predict_tree = cur_states[cur_step - 1]._stack[cur_states[cur_step - 1]._stack_size - 1].str

                        for sent, tags, type in zip(doc.origin_sentences, doc.sentences_tags, doc.sent_types):
                            for w, tag in zip(sent, tags):
                                outf.write(w + ' ')
                            outf.write(type[-1])
                            outf.write('\n')
                        for info in doc.other_infos:
                            outf.write(info + '\n')
                        outf.write(f'# newdoc id = {doc.tree_str}\n')
                        outf.write(predict_tree + '\n\n')
                except Exception as e:
                    print(f'[predict_batch] error: {e}')
                    for idx in range(len(onebatch)):
                        doc = onebatch[idx][0]
                        for sent, tags, type in zip(doc.origin_sentences, doc.sentences_tags, doc.sent_types):
                            for w, tag in zip(sent, tags):
                                outf.write(w + ' ')
                            outf.write(type[-1])
                            outf.write('\n')
                        for info in doc.other_infos:
                            outf.write(info + '\n')
                        outf.write(f'# newdoc id = {doc.tree_str}\n')
                        outf.write('{[(error)]}\n\n')

        end = time.time()
        print(f"Doc num: {len(data)},  parser time = {end - start:.2f}s")

    def construct_rst_data(self, edu_file, rst_file):
        with open(edu_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            with open(rst_file, 'w', encoding='utf-8') as f:
                print(f'write rst to {rst_file}')
                for k, v in data.items():
                    id = k
                    length = len(v['sentences'])
                    for sent in v['sentences']:
                        f.write(sent + '\n')
                    for i in range(length):
                        f.write('()\n')
                    f.write(f'{id}\n\n')
                    
    def start_parsing(self, edu_file):
        rst_file = edu_file + '.rst'
        self.construct_rst_data(edu_file=edu_file, rst_file=rst_file)
        data = inst(read_corpus(file_path=rst_file, edu_path=edu_file))
        self.predict_batch(data=data, output_file=rst_file+'.out')
        return rst_file + '.out'
