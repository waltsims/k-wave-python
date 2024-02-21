import os


def download_from_gdrive(file_id, output_path):
    try:
        import gdown
    except ModuleNotFoundError:
        raise AssertionError("This example requires `gdown` to be installed. " "Please install using `pip install gdown`")

    url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    gdown.download(url, output_path, quiet=False)


def download_if_does_not_exist(file_id, output_path):
    if not os.path.exists(output_path):
        download_from_gdrive(file_id, output_path)
