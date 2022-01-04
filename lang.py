class Lang:
    def __init__(self):
        #initialize containers to hold the words and corresponding index
        self.word2index = {'_SOS': 0, '_EOS':1 , '_PAD':2}
        self.word2count = {'_SOS': 0, '_EOS':0 , '_PAD':0}
        self.index2word = {0: '_SOS', 1: '_EOS' , 2: '_PAD'}
        self.n_words = 3

    #split a sentence into words and add it to the container
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    #If the word is not in the container, the word will be added to it,
    #else, update the word counter
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1