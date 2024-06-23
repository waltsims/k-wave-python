import importlib
import pytest
from unittest.mock import patch


@pytest.mark.parametrize("platform_name", ["Darwin"])
@pytest.mark.usefixtures("fs")
def test_not_implemented_error(platform_name):
    with pytest.raises(NotImplementedError):
        with patch("platform.system", return_value=platform_name):
            import kwave

            importlib.reload(kwave)


if __name__ == "__main__":
    pytest.main([__file__])
