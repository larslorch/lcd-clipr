from joblib import Memory

from lcd_clipr.definitions import SCRATCH_STORE_DIR, SUBDIR_JOBLIB


class NoMemory:
    @staticmethod
    def cache(func, *_, **__):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def clear():
        pass


def init_cache(ignore=False):
    if ignore:
        return NoMemory()
    else:
        return Memory(SCRATCH_STORE_DIR / SUBDIR_JOBLIB, verbose=0)

