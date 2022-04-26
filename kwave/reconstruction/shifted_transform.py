from uff import Position, Transform


class ShiftedTransform(Transform):
    def __call__(self, point):
        if isinstance(point, Position):
            # todo: add rotation
            return Position(point.x + self.translation.x, point.y + self.translation.y, point.z + self.translation.z)
        else:
            raise TypeError(f"Type {type(point)} not recognized.")
        pass
