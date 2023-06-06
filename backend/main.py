from src.utils import get_files_from_folder, open_txt


if __name__ == "__main__":
    books_folder = "books"
    books = get_files_from_folder(books_folder)
    print(open_txt(f"{books_folder}/{books[0]}"))
