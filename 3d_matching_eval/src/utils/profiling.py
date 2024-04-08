import cProfile
import logging
import pstats

from functools import wraps
from pathlib import Path
from typing import Union


def profile(
    output_file: Union[Path, str] = None,
    output_root: Union[Path, str] = None,
    sort_by: str = "cumulative",
    lines_to_print: bool = None,
    strip_dirs: bool = False,
    enabled: bool = True
):
    """A time profiler decorator.

    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/

    Args:
        output_file: Path or None. Default is None
            Path of the output file.
            If it's None, the name of the decorated function is used.
        output_root: Path or None. Default is None
            Path to the output dir. If the value is given, use it as the output
            directory for all profiling outputs.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
        enabled: bool
            Whether to enable profiler.
    Returns:
        Profile of the decorated function
    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            root_dir = Path(output_root or ".")
            if not root_dir.exists():
                root_dir.mkdir(parents=True)

            prof_file = Path(output_file or func.__name__ + ".prof")
            if not prof_file.is_absolute():
                prof_file = root_dir.joinpath(prof_file)
            prof_readable_file: Path = prof_file.parent.joinpath("readable__" + prof_file.name)

            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()

            pr.dump_stats(str(prof_file))
            pr.dump_stats(str(prof_readable_file))

            with open(str(prof_readable_file), "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval
        return wrapper
    return inner
