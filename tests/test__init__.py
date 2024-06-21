import pytest
import importlib
import tempfile
from unittest.mock import patch, MagicMock


@pytest.mark.parametrize("platform_name", ["Darwin"])
def test_not_implemented_error(platform_name):
    with pytest.raises(NotImplementedError):
        with patch("platform.system", return_value=platform_name):
            import kwave

            importlib.reload(kwave)


@pytest.fixture
def mock_windows_platform():
    with patch("platform.system", return_value="Windows") as mock:
        yield mock


@pytest.fixture
def mock_temp_file():
    with tempfile.NamedTemporaryFile() as temp_file:
        yield temp_file.name


def test_windows_dll_download_once(mock_windows_platform, mock_temp_file):
    # Mock the urllib.request.urlretrieve to return the path to the temporary file
    with patch("kwave.urllib.request.urlretrieve", return_value=(mock_temp_file, MagicMock())) as mock_urlretrieve:
        import kwave

        # Reload kwave and assert that urlretrieve is not called again
        importlib.reload(kwave)
        mock_urlretrieve.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
