import os


def get_files_from_folder(folder_path):
    """
    Get all files from a folder
    :param folder_path: path to the folder
    :return: list of files
    """
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)
    return files


def open_txt(file: str):
    """
    Open a txt file and return its content
    :param file: path to the file
    :return: content of the file
    """
    with open(file, "r") as f:
        return f.read()
