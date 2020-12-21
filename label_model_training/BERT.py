from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math

import numpy as np
import torch
import tqdm
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


class QA:

    def __init__(self, model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.model, self.tokenizer = self.load_model(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str, do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "/config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer

    def predict(self, passage: str, question: str):
        example = input_to_squad_example(passage, question)
        features = squad_examples_to_features(example, self.tokenizer, self.max_seq_length, self.doc_stride,
                                              self.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]
                          }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example, features, all_results, self.n_best_size, self.max_answer_length,
                            self.do_lower_case)
        return answer


if __name__ == '__main__':
    model = QA(model_path='./output')
    with open('./data/structured_hlm_new.json') as f:
        book = json.loads(f.read())
        labeled_book = []
        for section in tqdm.tqdm(book):
            labeled_section = dict()
            labeled_section['section_num'] = section['section_num']
            labeled_section['section_title'] = section['section_title']
            print(labeled_section['section_num'], labeled_section['section_title'])
            labeled_section['talks'] = []
            for talks in section['talks']:
                labeled_talks = []
                for talk in talks:
                    predict = model.predict(talk['context'], "说下一句话的人是谁？")
                    talk['speaker'] = predict['answer']
                    # print(talk)
                    labeled_talks.append(talk)
                labeled_section['talks'].append(labeled_talks)
            print(labeled_section)
            labeled_book.append(labeled_section)
        print('saving...')
        with open('./output/speaker_labeled_hlm_new.json', 'w+') as out_f:
            out_f.write(json.dumps(labeled_book, ensure_ascii=False))
