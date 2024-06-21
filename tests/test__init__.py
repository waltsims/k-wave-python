import pytest
import importlib
from unittest.mock import patch


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


def test_windows_dll_download_once(mock_windows_platform):
    def side_effect(url, filename):
        with open(filename, "wb") as f:
            f.write(b"Dummy file contents")

    # Mock the urllib.request.urlretrieve where it is used in the kwave module
    with patch("urllib.request.urlretrieve", side_effect=side_effect) as mock_urlretrieve:
        # First import to simulate initial download
        import kwave

        importlib.reload(kwave)

        # Assert urlretrieve was called during the initial import
        mock_urlretrieve.assert_called()

        # Reset mock to clear previous call history
        mock_urlretrieve.reset_mock()

        # Reload kwave to check if urlretrieve is not called again
        importlib.reload(kwave)

        # Assert urlretrieve was not called again
        mock_urlretrieve.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
