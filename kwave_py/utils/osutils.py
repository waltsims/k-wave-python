import platform


def is_unix():
    return platform.system() in ['Linux', 'Darwin']
