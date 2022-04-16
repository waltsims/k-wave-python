
class AddPositionIterator:

    def __init__(self, position):
        self._position = position
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._position):
            if self._index == 0:
                result = self._position.x
            elif self._index == 1:
                result = self._position.y
            elif self._index == 2:
                result = self._position.z
            self._index += 1
            return result
        else:
            raise StopIteration
