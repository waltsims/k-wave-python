import pytest
import importlib
from unittest.mock import patch


def test__init():
    with pytest.raises(NotImplementedError):
        with patch("platform.system", lambda: "Darwin"):
            importlib.reload(kwave)
