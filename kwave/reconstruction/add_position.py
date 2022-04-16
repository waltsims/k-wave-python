from uff import Position

from kwave.reconstruction.add_position_iterator import AddPositionIterator


class AddPosition(Position):

    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        self._index = 0

    def __len__(self):
        return 3

    def __add__(self, p2):
        if type(p2) in [Position, AddPosition]:
            return AddPosition(x=p2.x + self.x, y=p2.y + self.y, z=p2.z + self.z)
        else:
            return AddPosition(x=p2 + self.x, y=p2 + self.y, z=p2 + self.z)

    def __sub__(self, p2):
        if type(p2) in [Position, AddPosition]:
            return AddPosition(x=p2.x - self.x, y=p2.y - self.y, z=p2.z - self.z)
        else:
            return AddPosition(x=self.x - p2, y=self.y - p2, z=self.z - p2)

    def __truediv__(self, p2):
        if type(p2) in [Position, AddPosition]:
            return AddPosition(x=self.x / p2.x, y=self.y / p2.y, z=self.z / p2.z)
        else:
            return AddPosition(x=self.x / p2, y=self.y / p2, z=self.z / p2)

    def __mul__(self, p2):
        if type(p2) in [Position, AddPosition]:
            return AddPosition(x=p2.x * self.x, y=p2.y * self.y, z=p2.z * self.z)
        else:
            return AddPosition(x=p2 * self.x, y=p2 * self.y, z=p2 * self.z)

    def __floordiv__(self, p2):
        if type(p2) in [Position, AddPosition]:
            return AddPosition(x=self.x // p2.x, y=self.y // p2.y, z=self.z // p2.z)
        else:
            return AddPosition(x=self.x // p2, y=self.y // p2, z=self.z // p2)

    def __pow__(self, other):
        return AddPosition(x=self.x ** other, y=self.y ** other, z=self.z ** other)

    def __iter__(self):
        return AddPositionIterator(self)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise IndexError(f"Index {item} out of bounds for type {__name__}")
