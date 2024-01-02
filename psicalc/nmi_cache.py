class NmiCache:
    """
    Memoize NMI lookups
    """

    def __init__(self, f):
        self.cache = dict()
        self.nmi_func = f

    def get(self, a, b, msa):
        (a, b) = sorted([a, b])
        if a in self.cache:
            if b not in self.cache[a]:
                self.cache[a][b] = self.nmi_func(msa[:, a], msa[:, b])
        else:
            self.cache[a] = {b: self.nmi_func(msa[:, a], msa[:, b])}

        return self.cache[a][b]
