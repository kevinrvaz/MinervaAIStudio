import random
import string
import os


def create_random_file_name(ext):
    file = "".join(random.choices(string.ascii_letters, k=10))
    return os.path.join("generated_assets", f"{file}.{ext}")
