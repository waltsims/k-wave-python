from time import perf_counter


class TicToc(object):
    """
    A class for measuring the execution time of a code block.

    This class uses the perf_counter function from the time module to measure the
    execution time of a code block. It provides a simple interface with two methods:
    tic and toc. You can use the tic method to start the timer, and then use the
    toc method to stop the timer and get the elapsed time.
    """

    start_time = -1

    @staticmethod
    def tic():
        """
        Start the timer.

        This method sets the start_time attribute to the current time, as measured
        by the perf_counter function from the time module.
        """
        TicToc.start_time = perf_counter()

    @staticmethod
    def toc(reset: bool = False) -> float:
        """Stop the timer and return the elapsed time.

        This method calculates the elapsed time since the timer was started by
        subtracting the start_time attribute from the current time, as measured by
        the perf_counter function from the time module. If the reset argument is
        True, the timer will be restarted automatically.

        Args:
            reset: Whether to reset the timer after stopping it.

        Returns:
            The elapsed time in seconds.
        """
        passed_time = perf_counter() - TicToc.start_time
        if reset:
            TicToc.tic()
        return passed_time
