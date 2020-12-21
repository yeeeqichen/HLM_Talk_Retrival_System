import json


def split(row):
    '''split 25 ||| 28 ||| 李嬷嬷and return
    istart, iend, speaker'''
    tokens = row.split('|||')
    istart = int(tokens[0])
    iend = int(tokens[1])
    ## squad requires (-1, -1) if there is no answer
    if istart == -1:
        iend = -1
    speaker = tokens[2].strip()
    return [istart, iend, speaker]


def read_samples(context_file, result_file, title="红楼梦", flag=False):
    res = dict()
    res["title"] = title
    res["paragraphs"] = []
    with open(result_file, "r") as fin:
        lines = fin.readlines()
        labels = [split(line) for line in lines]
    with open(context_file, "r") as fin:
        contexts = fin.readlines()
    length = len(contexts)
    if flag:
        length = 36000
    for i in range(length):
        answer = {"answer_start": labels[i][0], "text": labels[i][2]}
        answers = [answer]
        para_entry = dict()
        para_entry["context"] = contexts[i]
        qas = [{"answers": answers,
            "question": "说下一句话的人是谁？",
            "id": i}]
        para_entry["qas"] = qas
        res["paragraphs"].append(para_entry)
    return res


def training_example(flag=False):
    res = read_samples("data/training_sentence.csv", "data/training_labels.csv", flag=flag)
    print(len(res['paragraphs']))
    out = {"data": [res], "version": "chinese_squad_v1.0"}
    file_name = "data/chinese_speaker_squad"
    if flag:
        file_name += '_small.json'
    else:
        file_name += '.json'
    with open(file_name, "w") as fout:
        fout.write(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    training_example(True)
