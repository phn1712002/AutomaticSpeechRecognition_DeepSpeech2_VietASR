import pandas as pd

class Dataset:
    def __init__(self, path):
        self.path = path

    def __call__(self, name_files):
        X, Y = [], []
        for file_name in name_files:
            # Đọc dữ liệu
            data = pd.read_csv(self.path + 'vi' + '/' + file_name + '.tsv', sep='\t')
            data = data[['path', 'sentence']]

            # Chèn thêm đầu path và lower sentence
            data['path'] = self.path + 'vi' + '/clips/' + data['path'].astype(str)
            data['sentence'] = data['sentence'].apply(lambda x: x.lower())

            tempX, tempY = list(data['path']), list(data['sentence'])
            X += tempX
            Y += tempY
        return X, Y
