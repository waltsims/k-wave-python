import pytest
from unittest.mock import patch


def test__init():
    with pytest.raises(NotImplementedError):
        with patch("platform.system", lambda: "Darwin"):
            import kwave  # noqa: F401
