from src.utils import get_files_from_folder, open_txt


def test_get_files_from_folder():
    assert get_files_from_folder("tests/data/books") == ["book1.txt", "book2.txt"]
