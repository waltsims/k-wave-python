# import os
# import importlib
# import pytest
# from unittest.mock import patch

# import kwave


# @pytest.mark.parametrize("platform_name", ["Darwin"])
# @pytest.mark.usefixtures("fs")
# def test_not_implemented_error(platform_name):
#     with pytest.raises(NotImplementedError):
#         with patch("platform.system", return_value=platform_name):
#             import kwave

#             importlib.reload(kwave)


# @pytest.fixture
# def mock_windows_platform():
#     with patch("platform.system", return_value="Windows") as mock:
#         yield mock


# @pytest.fixture
# def mock_download(fs):  # Use `fs` fixture from `pyfakefs`
#     created_files = []
#     fs.create_dir(kwave.BINARY_PATH)

#     def _create_file(url, file_path):
#         content = "Dummy content"
#         with open(file_path, "w") as f:
#             f.write(content)
#         created_files.append(file_path)
#         return file_path

#     yield _create_file

#     # Cleanup: Remove all created files (not necessary with pyfakefs, but kept for completeness)
#     for filepath in created_files:
#         try:
#             os.remove(filepath)
#         except FileNotFoundError:
#             pass


# @pytest.mark.usefixtures("fs")
# def test_windows_dll_download_once(mock_windows_platform, mock_download):  # Use `fs` fixture
#     # Patch the `urllib.request.urlretrieve` to use `mock_download`
#     with patch("urllib.request.urlretrieve", side_effect=mock_download) as mock_urlretrieve:
#         # First import to simulate initial download
#         import kwave

#         importlib.reload(kwave)

#         # Assert `urlretrieve` was called during the initial import
#         mock_urlretrieve.assert_called()

#         # Reset mock to clear previous call history
#         mock_urlretrieve.reset_mock()

#         # Reload `kwave` to check if `urlretrieve` is not called again
#         importlib.reload(kwave)

#         # Assert `urlretrieve` was not called again
#         mock_urlretrieve.assert_not_called()


# if __name__ == "__main__":
#     pytest.main([__file__])
