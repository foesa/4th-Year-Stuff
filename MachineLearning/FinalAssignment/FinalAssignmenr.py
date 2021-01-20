import json
import pandas as pd
from googletrans import Translator
from google_trans_new import google_translator
import os.path
from os import path


def read_file(text=None):
    if text is None:
        Y = []
        Z = []
        with open('reviews_262.jl.txt', encoding='utf8') as json_file:
            text_vals = []
            for i in json_file:
                data = json.loads(i)
                text_vals.append(data['text'])
                Y.append(data['voted_up'])
                Z.append(data['early_access'])
            return text_vals, Y, Z
    translations = []
    origins = []
    with open(text) as json_file:
        for i in json_file:
            data = json.loads(i)
            translations.append(data['text'])
            origins.append(data['origin'])
    return translations, origins


def translate_text(text):
    translator = Translator()
    translation = translator.translate(text)
    origin = text
    translated = translation
    return origin, translated


def write_translations(texts, origins):
    with open('translations.txt', 'w') as f:
        for text, origin in zip(texts, origins):
            f.write('{"text" : ' + '"' + text + '"' + '}' + '{"origin" : ' + '"' + origin + '"' + '}' + '\n')


def main():
    texts, Y, Z = read_file()
    origin = []
    translated = []
    if path.exists('translations.txt'):
        translated, origin = read_file('translations.txt')
    else:
        i = 0
        # chunks = [texts[x:x + 100] for x in range(0, len(texts), 100)]
        # for chunk in chunks:
        #     part_origin, part_translated = translate_text(chunk)
        #     print(part_translated)
        #     print(i)
        #     i = i +1
        for text in texts:
            part_origin, part_translated = translate_text(text)
            origin.append(part_origin)
            translated.append(part_translated)
            print(i, part_translated)
            i = i + 1
        write_translations(translated, origin)
    data = pd.DataFrame({'Origin': origin,
                         'Translated': translated,
                         'Voted Up': Y,
                         'Early Access': Z})
    print(data)


if __name__ == '__main__':
    main()

read_file()
