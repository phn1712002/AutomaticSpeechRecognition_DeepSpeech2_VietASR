import pandas as pd

class Dataset:
    def __init__(self, path):
        self.path = path

    def __call__(self, name_files):

        def split(string, pathFolder):
            index = string.find(" ")
            path = string[0:index]
            sentence = string[index + 1: len(string)]

            index = path.find("_")
            path = pathFolder + path[0:index] + '/' + path + '.wav'
            return path, sentence

        # Đọc dữ liệu
        X, Y = [], []
        for name_file in name_files:
            data = pd.read_csv(self.path + 'vi' + '/' + name_file + '/prompts.txt', header=None)
            path = self.path + 'vi' + '/' + name_file + '/waves/'
            # Cắt path và sentence
            listPath=[]
            listSentence=[]
            for index in range(0, len(data) - 1):
                path_full, sentence = split(data[0][index], path)
                listPath.append(path_full)
                listSentence.append(sentence)

            data = pd.DataFrame({'path': listPath,
                            'sentence': listSentence
                            })
            # lower label
            data['sentence'] = data['sentence'].apply(lambda x: x.lower())
            tempX, tempY = list(data['path']), list(data['sentence'])
            X += tempX
            Y += tempY
        return X, Y
