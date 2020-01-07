class Vocab(object):
    def __init__(self, filename=None, data=None):
        self.id2lab = {}
        self.label2id = {}

        # special label
        self.special = []

        if data is not None:
            self.addSpecials(data)
        if filename is not None:
            self.loadfiles(filename)

    def size(self):
        return len(self.id2lab)

    def loadfiles(self, file):
        for line in open(file).readlines():
            self.add(line.strip())

    def add(self, label):
        if label in self.label2id:
            idx = self.label2id[label]
        else:
            idx = self.size()
            self.label2id[label] = idx
            self.id2lab[idx] = label
        return idx

    def addSpecial(self, label, idx=None):
        idx = self.add(label)
        self.special += [idx]

    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    def getIndex(self, label, default=None):
        try:
            return self.label2id[label]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.id2lab[idx]
        except KeyError:
            return default

    def convertToId(self, labels, unk, bos=None, eos=None):
        vec = []
        if bos is not None:
            vec += [self.getIndex(bos)]
        unkid = self.getIndex(unk)
        vec += [self.getIndex(label, default=unkid) for label in labels]

        if eos is not None:
            vec += [self.getIndex(eos)]
        return vec

    def convertToLabel(self, idx, stop):
        labels = []
        for i in idx:
            labels.append(self.getLabel(idx))
            if i == stop:
                break
        return labels

    def export(self, path):
        with open(path, "w") as file:
            for label in self.label2id:
                file.write(label + "\n")
