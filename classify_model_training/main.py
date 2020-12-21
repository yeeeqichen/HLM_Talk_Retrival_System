import bert
import torch


config = bert.Config()
model = bert.Model(config)
model.to(config.device)
model.load_state_dict(torch.load(config.save_path))
model.eval()
top_N = 5


def to_bert_input(talk):
    pad_size = config.pad_size
    token = config.tokenizer.tokenize(talk)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    token_ids = torch.LongTensor([token_ids]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return (token_ids, seq_len, mask)


def predict(talk):
    bert_input = to_bert_input(talk)
    prob = torch.nn.functional.softmax(model(bert_input), dim=1).cpu()[0]
    ranks = torch.argsort(prob, descending=True).numpy()
    prob = prob.detach().numpy()
    max_prob = (config.class_list[ranks[0]], prob[ranks[0]])
    min_prob = (config.class_list[ranks[-1]], prob[ranks[-1]])
    top_n_prob = []
    for i in range(top_N):
        top_n_prob.append((config.class_list[ranks[i]], prob[ranks[i]]))
    return max_prob, min_prob, top_n_prob


if __name__ == '__main__':
    talks = ['好妹妹，你昨儿告了我了没有？叫我悬了一夜的心。', '下流东西们，我素日担待你们得了意，一点儿也不怕，越发拿着我取笑儿了', '我今年七十三了。',
             '我劝你耗子尾汁', '我要挂科了', '哎呀，打扫一个500平米的客厅真是太累了', '哎呀呀，我好漂亮', '你这小丫头片子，好大的胆子！', '这大观园里我说了算',
             '夫人，我再也不敢了，饶了我吧！', '我是葬花的主人公']
    for talk in talks:
        print(talk, predict(talk))

