import math

EPSILON = 0


class NmiCache:
    """
    Memoize NMI lookups
    """

    def __init__(self, f, _use_esp=False):
        self.cache = dict()
        self.nmi_func = lambda a, b: NmiValue(f(a, b, _use_esp=_use_esp)) if _use_esp else f(a, b, _use_esp=_use_esp)
        self._use_esp = _use_esp

    def get(self, a, b, msa):
        if self._use_esp:  # XXX: Temp for testing
            (a, b) = sorted([a, b])

        if a in self.cache:
            if b not in self.cache[a]:
                self.cache[a][b] = self.nmi_func(msa[:, a], msa[:, b])
        else:
            self.cache[a] = {b: self.nmi_func(msa[:, a], msa[:, b])}

        return self.cache[a][b]

    def clear(self):
        self.cache = dict()


class NmiValue:
    """
    Simple wrapper around NMI values to factor EPSILON into comparisons
    """

    def __init__(self, v):
        self.value = v

    def __eq__(self, o):
        if isinstance(o, NmiValue):
            return math.isclose(self.value, o.value, rel_tol=EPSILON)
        return math.isclose(self.value, o, rel_tol=EPSILON)

    def __gt__(self, o):
        if isinstance(o, NmiValue):
            return self.value > o.value and not math.isclose(self.value, o.value, rel_tol=EPSILON)
        return self.value > o and not math.isclose(self.value, o, rel_tol=EPSILON)

    def __ge__(self, o):
        if isinstance(o, NmiValue):
            return self.value > o.value or math.isclose(self.value, o.value, rel_tol=EPSILON)
        return self.value > o or math.isclose(self.value, o, rel_tol=EPSILON)

    def __lt__(self, o):
        if isinstance(o, NmiValue):
            return self.value < o.value and not math.isclose(self.value, o.value, rel_tol=EPSILON)
        return self.value < o and not math.isclose(self.value, o, rel_tol=EPSILON)

    def __le__(self, o):
        if isinstance(o, NmiValue):
            return self.value < o.value or math.isclose(self.value, o.value, rel_tol=EPSILON)
        return self.value < o or math.isclose(self.value, o, rel_tol=EPSILON)

    def __add__(self, o):
        return NmiValue(self.value + o.value if isinstance(o, NmiValue) else self.value + o)

    def __truediv__(self, o):
        return NmiValue(self.value / o.value if isinstance(o, NmiValue) else self.value / o)

    def __round__(self, n=0):
        return NmiValue(round(self.value, n))

    def __repr__(self):
        return str(self.value)
