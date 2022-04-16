from uff import Position, Transform
from kwave.reconstruction.add_position import AddPosition


class ShiftedTransform(Transform):
    def __call__(self, point):
        if type(point) in [AddPosition, Position]:
            # todo: add rotation
            return Position(point.x + self.translation.x, point.y + self.translation.y, point.z + self.translation.z)
        else:
            raise TypeError(f"Type {type(point)} not recognized.")
        pass
