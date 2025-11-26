import random
import string


def create_random_file_name(ext):
    file = "".join(random.choices(string.ascii_letters, k=10))
    return f"{file}.{ext}"
