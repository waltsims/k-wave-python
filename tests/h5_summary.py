import json
import logging
import os
from copy import deepcopy
from hashlib import sha256

import h5py
import numpy as np
from deepdiff import DeepDiff


class H5Summary(object):

    def __init__(self, summary: dict):
        self.summary = summary

    @staticmethod
    def from_h5(h5_path):
        data = {}

        def extract_summary(name, obj):
            np_obj = np.array(obj)
            data[name] = {
                'dtype': str(obj.dtype),
                'attrs': H5Summary._convert_attrs(obj.attrs),
                'shape': list(obj.shape),
                'checksums': {
                    # why to have " + 0" here?
                    # because in NumPy, there could be 0 & -0
                    # while hashing we only want single type of zero
                    # therefore we add 0 to have only non-negative zero
                    str(sd): sha256(np_obj.round(sd) + 0).hexdigest() for sd in range(6, 17, 2)
                }
            }

        with h5py.File(h5_path, 'r') as hf:
            hf.visititems(extract_summary)
            data['root'] = {
                'attrs': H5Summary._convert_attrs(hf.attrs)
            }
        return H5Summary(data)

    def save(self, name: str):
        path = self._get_json_filepath(name)

        assert not os.path.exists(path), 'File already exists, would not overwrite!'

        with open(path, "w") as write_file:
            json.dump(self.summary, write_file, indent=4)

    @staticmethod
    def load(name: str):
        path = H5Summary._get_json_filepath(name)
        with open(path, "r") as json_file:
            summary = json.load(json_file)
            return H5Summary(summary)

    @staticmethod
    def _get_json_filepath(name):
        if not name.endswith('.json'):
            name = name + '.json'
        return os.path.join('tests', 'reference_outputs', name)

    @staticmethod
    def _convert_attrs(attrs):
        return {k: str(v) for k, v in dict(attrs).items()}

    def get_diff(self, other, eps=1e-8, precision=8):
        assert isinstance(other, H5Summary)
        excluded = [
            "root['root']['attrs']['created_by']",
            "root['root']['attrs']['creation_date']",
            "root['root']['attrs']['file_description']",
            "root['Nt']"  # Skip Nt after updating kgrid logic
        ]
        own_summary = self._strip_checksums(precision)
        other_summary = other._strip_checksums(precision)
        diff = DeepDiff(own_summary, other_summary, exclude_paths=excluded, math_epsilon=eps)
        return diff

    def _strip_checksums(self, target_precision):
        if not isinstance(target_precision, str):
            target_precision = str(target_precision)

        summary = deepcopy(self.summary)
        for k in summary.keys():
            if 'checksums' in summary[k]:
                summary[k]['checksums'] = {target_precision: summary[k]['checksums'][target_precision]}
        return summary


if __name__ == '__main__':
    # h5_path = f'/private/var/folders/wd/pzn3h1fn37s6gbt12tyj50gw0000gn/T/example_input.h5'
    # summary = H5Summary.from_h5(h5_path)
    # summary.save(f'example_pr_2D_FFT_line_sensor')

    for i in range(1, 3):
        logging.log(logging.INFO, 'Processing file:', i)
        h5_path = f'/private/var/folders/wd/pzn3h1fn37s6gbt12tyj50gw0000gn/T/example_input_{i}.h5'
        folder = 'example_pr_2D_TR_directional_sensors'
        folder_path = os.path.join('tests', 'reference_outputs', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        summary = H5Summary.from_h5(h5_path)
        summary.save(f'{folder}/input_{i}')
