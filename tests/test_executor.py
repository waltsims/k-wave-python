import sys
import unittest.mock
import logging
from kwave.executor import Executor
import pytest
import os

check_is_linux = pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Currently only implemented for linux.")


@check_is_linux
@pytest.mark.skipif(os.environ.get("CI") == 'true', reason="Running in GitHub Workflow.")
class TestExecutor(unittest.TestCase):

    @unittest.mock.patch('os.system')
    def test_system_call_correct(self, os_system):
        try:
            ex = Executor('cpu')
            input_filename = '/tmp/input.h5'
            output_filename = '/tmp/output.h5'
            try:
                ex.run_simulation(input_filename, output_filename, options='')
            except OSError:
                logging.info("Caught 'Unable to open file' exception.")

            call_str = f"{ex.binary_path} -i {input_filename} -o {output_filename} "
            os_system.assert_called_once_with(call_str)
        except NotImplementedError as err:
            if not sys.platform.startswith('darwin'):
                raise err


if __name__ == '__main__':
    unittest.main()
