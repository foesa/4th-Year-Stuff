import json
import pandas as pd
from googletrans import Translator

def read_file():
    X = []
    Y = []
    Z = []
    with open('reviews_262.jl.txt', encoding='utf8') as json_file:
        translator = Translator()
        for i in json_file:
            data = json.loads(i)
            if isEnglish(data['text']):
                X.append(data['text'])
                Y.append(data['voted_up'])
                Z.append(data['early_access'])
            else:
                translation = translator.translate(data['text'])
                print(translation)

    print(len(X))
    dataFrame = pd.DataFrame(data={'Text': X, 'Voted Up': Y, 'Early Access': Z})
    print(dataFrame)
    return dataFrame


def isEnglish(text) -> bool:
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


read_file()
