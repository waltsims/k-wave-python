

def download_from_gdrive(file_id, output_path):

    import gdown

    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
