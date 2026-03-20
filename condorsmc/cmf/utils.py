import shutil


def archive_condorcmf(path, archive_path):
    """
    Archives a directory.

    Args:
        path (str): Path to the directory to be archived.
        archive_path (str): Path to the archive file.
    """

    shutil.make_archive(archive_path, "zip", path)
