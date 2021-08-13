from time import perf_counter


class TicToc(object):
    start_time = -1

    @staticmethod
    def tic():
        TicToc.start_time = perf_counter()

    @staticmethod
    def toc(reset: bool = False):
        passed_time = perf_counter() - TicToc.start_time
        if reset:
            TicToc.tic()
        return passed_time