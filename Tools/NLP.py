from keras.layers import StringLookup

class Tokenization():
    def __init__(self):
        self.vocabulary = None
        self.char_to_num = None
        self.num_to_char = None

    def _createList(self, vocabulary=None):
        self.vocabulary = vocabulary
        self.char_to_num = StringLookup(vocabulary=vocabulary, oov_token='UNK')
        self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token='UNK', invert=True)
        return self
    
    def settingWithList(self, vocabulary=None):
        return self._createList(vocabulary)
    
    def getVocabulary(self):   
        return self.vocabulary
    
    def getLenVocabulary(self):
        return self.char_to_num.vocabulary_size()