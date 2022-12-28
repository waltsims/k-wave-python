from typing import Any


class dotdict(dict):
    """
    A dictionary supporting dot notation.

    This class extends the built-in `dict` type by adding support for accessing
    items using dot notation (e.g. `dotdict.a.b.c`) instead of square bracket
    notation (e.g. `dotdict['a']['b']['c']`). The class also provides a
    `lookup` method for looking up a value in a nested dictionary structure
    using a dot-separated path (e.g. "a.b.c").

    Examples:
        >>> d = dotdict({'a': {'b': {'c': 1}}})
        >>> d.a.b.c
        1
        >>> d.lookup('a.b.c')
        1

    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def lookup(self, dotkey: str) -> Any:
        """
        Look up a value in a nested dictionary structure using a dot-separated path.

        Args:
            dotkey: A dot-separated path to the value, e.g. "a.b.c".

        Returns:
            The value at the specified path.

        Raises:
            KeyError: If the specified path does not exist in the dictionary.

        """
        path = list(reversed(dotkey.split(".")))
        v = self
        while path:
            key = path.pop()
            if isinstance(v, dict):
                v = v[key]
            elif isinstance(v, list):
                v = v[int(key)]
            else:
                raise KeyError(key)
        return v
