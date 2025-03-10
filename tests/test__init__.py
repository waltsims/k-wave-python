import importlib
from unittest.mock import patch

import pytest


def test__init():
    with pytest.raises(NotImplementedError):
        with patch("platform.system", lambda: "Unknown"):
            import kwave

            importlib.reload(kwave)


if __name__ == "__main__":
    pytest.main([__file__])
