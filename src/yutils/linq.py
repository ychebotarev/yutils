import itertools

class Iteratable():
    def __init__(self, iteratable):
        if not hasattr(iteratable, "__iter__"):
            raise TypeError("Enumerable must be instantiated with an iterable object")    
        self._iteratable = iteratable

    def __iter__(self):
        return iter(self._iteratable)

    def __getitem__(self, n):
        for i, e in enumerate(self):
            if i == n:
                return e        

    def select(self, func):
        return SelectGenerator(iter(self), func)

    def where(self, predicate):
        if predicate is None:
            raise ValueError("No predicate given for where clause")        
        return WhereGenerator(iter(self), predicate)

    def count(self):
        return sum(1 for element in self)        

    def count_if(self, predicate=None):
        if predicate is not None:
            return sum(1 for element in self.where(predicate))
        return sum(1 for element in self)        

    def first(self, predicate=None):
        result = self.first_or_none(predicate)
        if result is None:
            raise IndexError
        return result

    def first_or_none(self, predicate=None):
        result = self[0] if (predicate is  None) else self.where(predicate)[0]
        return result

    def last(self, predicate=None):
        result = self.last_or_none(predicate)
        if result is None:
            raise IndexError
        return result

    def last_or_none(self, predicate=None):
        result = None
        if predicate is not None:
            for e in self.where(predicate):
                result = e    
        else:
            for e in iter(self):
                result = e    
        return result

    def to_list(self):
        return [item for item in iter(self)]
    
    def for_each(self, func):
        if func is None:
            raise ValueError("No func given for for_each clause")        
        for e in iter(self):
            func(e)
    
    def take(self, n):
        return Iteratable(itertools.islice(iter(self), n))
    
    def takewhile(self, predicate):
        if predicate is None:
            raise ValueError("No predicate given for takewhile clause")        
        return Iteratable(itertools.takewhile(predicate, iter(self)))

    def skip(self, n):
        return SkipGenerator(iter(self), n)
    
    def skipwhile(self, predicate):
        if predicate is None:
            raise ValueError("No predicate given for skipwhile clause")        
        return Iteratable(itertools.dropwhile(predicate, iter(self)))

    def select_many(self, func=lambda x: x):
        return SelectManyGenerator((iter(self)), func)
    
    def zip(self, iteratable):
        return ZipGenerator((iter(self)), iteratable)

class SelectGenerator(Iteratable):
    def __init__(self, iteratable, func):
        super(SelectGenerator, self).__init__(iteratable)
        self.func = func

    def __iter__(self):
        for e in iter(self._iteratable):
            yield self.func(e)

class SelectManyGenerator(Iteratable):
    def __init__(self, iteratable, func):
        super(SelectManyGenerator, self).__init__(iteratable)
        self.func = func

    def __iter__(self):
        for e in iter(self._iteratable):
            collection = self.func(e)
            for c in iter(collection):
                yield c

class WhereGenerator(Iteratable):
    def __init__(self, iteratable, predicate):
        super(WhereGenerator, self).__init__(iteratable)
        self.predicate = predicate

    def __iter__(self):
        for e in iter(self._iteratable):
            if self.predicate(e):
                yield e

class SkipGenerator(Iteratable):
    def __init__(self, iteratable, skip):
        super(SkipGenerator, self).__init__(iteratable)
        self.skip = skip

    def __iter__(self):
        i = 0
        for e in iter(self._iteratable):
            if i>=self.skip:
                yield e
            i+=1

class ZipGenerator(Iteratable):
    def __init__(self, iteratable1, iteratable2):
        super(ZipGenerator, self).__init__(iteratable1)
        self.iteratable2 = iteratable2
    
    def __iter__(self):
        return zip(iter(self._iteratable), iter(self.iteratable2))