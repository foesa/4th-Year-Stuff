import json
import re
from os import path
import pandas as pd
from google_trans_new import google_translator
from FinalAssignment.CNN import main as cnn
from FinalAssignment.pre_proceess import main as pre_process


def read_file(text=None):
    if text is None:
        Y = []
        Z = []
        with open('../reviews_262.jl.txt', encoding='utf8') as json_file:
            text_vals = []
            for i in json_file:
                data = json.loads(i)
                text_vals.append(data['text'])
                Y.append(data['voted_up'])
                Z.append(data['early_access'])
            return text_vals, Y, Z
    translations = []
    with open(text, encoding='utf-8') as json_file:
        for i in json_file:
            data = json.loads(i)
            try:
                translations.append(data['text'])
            except TypeError:
                return pd.DataFrame(json.loads(i))
    return translations


def fixFile():
    with open('translations.txt', 'r', encoding="utf-8") as s:
        with open('../new_translations.txt', 'a', encoding="utf-8") as f:
            for i in s:
                matches = re.findall(r'\"(.+?)\"', i)
                matches.pop(0)
                f.write('{"text" : ' + '"' + ",".join(matches) + '"' + '}' + '\n')


def translate_text(text):
    translator = google_translator()
    translation = translator.translate(text)
    origin = text
    translated = translation
    return origin, translated


def write_translations(texts):
    with open('translations.txt', 'a', encoding="utf-8") as f:
        for text in texts:
            f.write('{"text" : ' + '"' + text + '"' + '}' + '\n')


def main():
    texts, Y, Z = read_file()
    translated = []
    if path.exists('../new_translations.txt'):
        # fixFile()
        translated = read_file('new_translations.txt')
    else:
        i = 0
        for text in texts:
            if i >= 4643:
                part_origin, part_translated = translate_text(text)
                translated.append(part_translated)
                write_translations(translated)
                translated = []
            i = i + 1
    dataset = pd.DataFrame({'Text': translated,
                            'Voted Up': Y,
                            'Early Access': Z})
    if not path.exists('../processed_reviews.txt'):
        pre_process(dataset['Text'])
    else:
        tokens = read_file('processed_reviews.txt')
        dataset["tokens"] = tokens
    print(dataset.loc[0])
    # svc(dataset)
    print(dataset.groupby('Early Access').count())
    cnn(dataset)


if __name__ == '__main__':
    main()

read_file()
