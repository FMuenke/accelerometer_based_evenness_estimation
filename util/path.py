import os


def check_n_make_dir(tar_dir, clean=False):
    """
    checks if a directory exits and maks one if necessary
    :param tar_dir:
    :param clean: if True all files in folder will be deleted
    :return:
    """
    if not os.path.isdir(tar_dir):
        os.mkdir(tar_dir)

    if clean:
        for f in os.listdir(tar_dir):
            if not os.path.isdir(os.path.join(tar_dir, f)):
                os.remove(os.path.join(tar_dir, f))
            else:
                check_n_make_dir(os.path.join(tar_dir, f), clean=True)
