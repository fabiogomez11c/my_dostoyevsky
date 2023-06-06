from src.utils import get_files_from_folder, open_txt


def test_get_files_from_folder():
    assert get_files_from_folder("tests/data/books") == ["book1.txt", "book2.txt"]


def test_open_txt():
    assert (
        open_txt("tests/data/books/book1.txt")
        == "This is a test file.\nIt is used for testing.\n"
    )
