from tqdm import tqdm_notebook
import re
import json


def get_conversations(sentence):
    end_symbols = ['"', '“', '”']
    istart, iend = -1, -1
    talks = []
    for i in range(1, len(sentence)):
        if (not istart == -1) and sentence[i] in end_symbols:
            iend = i
            conversation = {'istart':istart, 'iend':iend, 'talk':sentence[istart+1:iend]}
            talks.append(conversation)
            istart = -1
        if sentence[i-1] in [':', '：'] and sentence[i] in end_symbols:
            istart = i
    contexts = []
    if len(talks):
        for i in range(len(talks)):
            if i == 0:
                contexts.append(sentence[:talks[i]['istart']])
            else:
                contexts.append(sentence[talks[i-1]['iend']+1:talks[i]['istart']])
        if talks[-1]['iend'] != len(sentence):
            contexts.append(sentence[talks[-1]['iend']+1:])
        else:
            contexts.append(' ')
        for i in range(len(talks)):
            talks[i]['context'] = contexts[i]

    return talks, contexts


def main():
    cnt = 0
    book = []
    with open('data/红楼梦前80.txt') as f:
        section = {}
        conversation_num = 0
        total_conversation_num = 0
        for line in tqdm_notebook(f.readlines()):
            paragraph = line.strip().replace(' ', '')
            match = re.match('(第[0-9]+章)(.*)', paragraph)
            if match:
                if cnt:
                    total_conversation_num += conversation_num
                    print('section_num: {}, section_title: {}, section_conversations: {}'.
                          format(section['section_num'], section['section_title'], conversation_num))
                    book.append(section)
                    section = {}
                    conversation_num = 0
                # print(groups.groups())
                print(match.groups())
                cnt += 1
                section['section_num'] = match.groups()[0]
                section['section_title'] = match.groups()[1]
                section['talks'] = []
                continue
            talks, contexts = get_conversations(line.strip().replace(' ', ''))
            if len(talks):
                section['talks'].append(talks)
                conversation_num += len(talks)
        total_conversation_num += conversation_num
        print('section_num: {}, section_title: {}, section_conversations: {}'.
              format(section['section_num'], section['section_title'], conversation_num))
        book.append(section)
        print('section_counts: ', cnt)
        print('total_conversations: ', total_conversation_num)
        print(len(book))
    save_structured_data(book)


def save_structured_data(book):
    with open('data/structured_hlm_new.json', 'w+') as f:
        f.write(json.dumps(book, ensure_ascii=False))


if __name__ == '__main__':
    main()