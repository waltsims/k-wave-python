from dataclasses import dataclass
from typing import List, Dict
import matlab.engine
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy.io import loadmat, savemat
from pathlib import Path

from converter.kspaceFirstOrder2D_543_563 import calculate_rho0_sgx_sgy
from kwave.utils.dotdictionary import dotdict


@dataclass
class CodeBlock:
    filename: str
    start_line: int
    end_line: int

    def extract_as_a_file(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()
        lines = lines[self.start_line - 1:self.end_line]
        lines = [line for line in lines if not line.lstrip().startswith("%")]
        block = "".join(lines)
        # write to a file
        output_filename = "script.m"
        with open(output_filename, "w") as f:
            f.write(block)
        return output_filename


class Lifecycle(object):

    def __init__(self):
        self.eng = self.init_matlab_engine()

    def init_matlab_engine(self):
        self.eng = matlab.engine.start_matlab()
        self.populate_matlab_path(self.eng)
        return self.eng

    @staticmethod
    def populate_matlab_path(eng):
        eng.addpath(eng.genpath('/Users/farid/workspace/k-wave-toolbox-version-1.4'), nargout=0)
        eng.addpath(eng.genpath('/Users/farid/workspace/k-wave-toolbox-version-1.3-cpp-linux-binaries'), nargout=0)
        eng.addpath(eng.genpath('/Users/farid/workspace/uff_030'), nargout=0)

    def push_variables_to_matlab_workspace(self, variables: Dict):
        mapped_vars = {}
        for key in list(variables.keys()):
            if '.' in key:
                new_key = key.replace('.', '_')
                mapped_vars[new_key] = key
                variables[new_key] = variables.pop(key)

        savemat("workspace_pre.mat", variables)
        self.eng.eval("load('workspace_pre.mat')", nargout=0)
        for new_key, old_key in mapped_vars.items():
            self.eng.eval(f"{old_key}={new_key};", nargout=0)
            self.eng.eval(f"clear {new_key};", nargout=0)

    def pull_variables_from_matlab_workspace(self):
        self.eng.eval("save('workspace_post.mat')", nargout=0)
        workspace_post = loadmat("workspace_post.mat", simplify_cells=True)
        return workspace_post

    def run_script(self):
        """
            Runs the Matlab code block and returns the variables in the workspace
        """
        self.eng.eval("script", nargout=0)
        return self.pull_variables_from_matlab_workspace()


lifecycle = Lifecycle()


@settings(max_examples=10)
@given(rho0=arrays(np.float64, (20, 30)))
def test_else_case(rho0):
    use_sg = False
    lifecycle.push_variables_to_matlab_workspace({"rho0": rho0, "flags.use_sg": use_sg})

    code_block = CodeBlock(
        filename="/Users/farid/workspace/k-wave-toolbox-version-1.3/k-Wave/kspaceFirstOrder2D.m",
        start_line=543,
        end_line=563
    )
    code_block.extract_as_a_file()
    workspace_post = lifecycle.run_script()

    rho0_sgx, rho0_sgy, rho0_sgx_inv, rho0_sgy_inv = calculate_rho0_sgx_sgy(
        rho0, None, workspace_post['flags'], np.shape)

    assert np.allclose(rho0_sgx, workspace_post['rho0_sgx'])
    assert np.allclose(rho0_sgy, workspace_post['rho0_sgy'])
    assert np.allclose(rho0_sgx_inv, workspace_post['rho0_sgx_inv'])
    assert np.allclose(rho0_sgy_inv, workspace_post['rho0_sgy_inv'])
    lifecycle.eng.clear(nargout=0)


if __name__ == '__main__':
    test_else_case()
