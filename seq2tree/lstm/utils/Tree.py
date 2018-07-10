class Tree:

    def __init__(self):
        self.parent = None
        self.num_childern = 0
        self.childern = []


    def add_child(self, c):
        if type(c) == Tree:
            c.parent = self

        self.childern.append(c)
        self.num_childern = self.num_childern + 1


    def size(self):
        if self.size != None:
            return self.size

        size = 1
        for c in self.childern:
            if type(c) == Tree:
                size += c.size()
            else:
                size += 1

        self.size = size

        return size
