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
    with open(text) as json_file:
        for i in json_file:
            data = json.loads(i)
            translations.append(data['text'])
    return translations


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
    origin = []
    translated = []
    if path.exists('translations.txt'):
        translated = read_file('translations.txt')
    else:
        i = 0
        for text in texts:
            if i >= 4643:
                part_origin, part_translated = translate_text(text)
                origin.append(part_origin)
                translated.append(part_translated)
                print(i, part_translated)
                write_translations(translated)
                translated = []
                origin = []
            i = i+1
            print(i)
    dataset = pd.DataFrame()


if __name__ == '__main__':
    main()

read_file()
