class SymbolManager:

    def __init__(self, vocab_path, min_freq=1):

        self.SOS = "<s>"
        self.EOS = "</s>"
        self.NT = "</n>"
        self.LF_B = "("
        self.RT_B = ")"
        self.UNK = "<unk>"

        self.id2symbol, self.symbol2id = self._create_vocab(vocab_path, min_freq)


    def _create_vocab(self, vocab_path, min_freq):

        vocab_f = open(vocab_path, "r", encoding="utf-8")

        vocab_data = vocab_f.readlines()
        vocab_f.close()

        id2symbol = [self.EOS, self.SOS, self.NT, self.UNK]
        symbol2id = {self.EOS:0, self.SOS:1, self.NT:2, self.UNK:3}

        for v in vocab_data:
            v = v.strip().split('\t')
            word = v[0].strip()
            freq = int(v[1])

            if freq >= min_freq:

                id2symbol.append(word)
                symbol2id[word] = len(id2symbol)-1


        return id2symbol, symbol2id

    def get_symbol2id(self, symbol):

        if symbol in self.symbol2id:
            return self.symbol2id[symbol]

        return self.symbol2id[self.UNK]


    def get_id2symbol(self, id):
        return self.id2symbol[id]

    def symbols2ids_list(self, symbol_list):
        l = []
        for symbol in symbol_list:
            l.append(self.get_symbol2id(symbol))

        return l

    def ids2symbols_list(self, id_list):
        l = []
        for id in id_list:
            l.append(self.get_id2symbol(id))

        return l

    def size(self):
        return len(self.symbol2id)
