from tqdm import tqdm
from concurrent import futures
import cv2


def pmap(f, iterable, max_threads=None, show_pbar=False, **kwargs):
    """Concurrent version of map()."""
    with futures.ThreadPoolExecutor(max_threads) as executor:
        if show_pbar:
            results = tqdm(executor.map(f, iterable, **kwargs))
        else:
            results = executor.map(f, iterable, **kwargs)
        return list(results)


def read_rgb_255(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def read_gray_255(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def read_mask(path, threshold=128):
    gray = read_gray_255(path)
    mask = gray > threshold
    return mask
