# -- coding: utf-8 --

"""Logger.py: A simple logging facade used for level-based printing."""

import sys
from typing import Callable, List
from tqdm import tqdm


class Logger:
    """simple logging facade for level-based printing."""
    # define logging levels
    LOG_LEVEL_RANGE: List[int] = range(4)
    MODE_SILENT, MODE_NORMAL, MODE_VERBOSE, MODE_DEBUG = LOG_LEVEL_RANGE

    @classmethod
    def setMode(cls, lvl: int) -> None:
        """Sets logging mode defining which message types will be printed."""
        cls.logProgressBar, cls.log, cls.logError, cls.logInfo, cls.logWarning, cls.logDebug = cls._fgen(
            cls.MODE_NORMAL if lvl not in cls.LOG_LEVEL_RANGE
            else lvl, cls.MODE_NORMAL, cls.MODE_VERBOSE, cls.MODE_DEBUG
        )

    @staticmethod
    def _fgen(lvl: int, MODE_NORMAL: int, MODE_VERBOSE: int, MODE_DEBUG: int) -> List[Callable]:
        """Composes lambda print functions for each logging level."""
        m_data = zip(
            [f'\033[{n}m\033[1m{m}\033[0m\033[0m: ' for n, m in zip(
                (91, 92, 93, 94),
                ('ERROR', 'INFO', 'WARNING', 'DEBUG')
            )],
            [MODE_VERBOSE, MODE_VERBOSE, MODE_VERBOSE, MODE_DEBUG]
        )
        return [
            (lambda iterable, **kwargs: tqdm(iterable, file=sys.stdout, dynamic_ncols=True, **kwargs)) if lvl >= MODE_NORMAL else (lambda iterable, **_: iterable),
            (lambda msg: tqdm.write(f'\033[1m{msg}\033[0m', file=sys.stdout)) if lvl >= MODE_NORMAL else (lambda _: None)
        ] + [
            (lambda msg, m_type=n: tqdm.write(f'{m_type}{msg}', file=sys.stdout)) if lvl >= m else (lambda _: None)
            for n, m in m_data
        ]

    # initialize to MODE_NORMAL
    logProgressBar, log, logError, logInfo, logWarning, logDebug = _fgen.__func__(
        MODE_NORMAL, MODE_NORMAL, MODE_VERBOSE, MODE_DEBUG
    )
