from tqdm import tqdm
from concurrent import futures


def pmap(f, iterable, max_threads=None, show_pbar=False, **kwargs):
    """Concurrent version of map()."""
    with futures.ThreadPoolExecutor(max_threads) as executor:
        if show_pbar:
            results = tqdm(executor.map(f, iterable, **kwargs))
        else:
            results = executor.map(f, iterable, **kwargs)
        return list(results)
