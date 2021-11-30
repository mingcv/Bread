import datetime
import os
import shutil
import time


def contains(key, lst):
    flag = False
    for item in lst:
        if key == item:
            flag = True
    return flag


def make_empty_dir(new_dir):
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)


def get_timestamp():
    return str(time.time()).replace('.', '')


def get_formatted_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__':
    pass
