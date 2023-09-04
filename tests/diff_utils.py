import logging
from pprint import pprint
from tests.h5_summary import H5Summary


def compare_against_ref(reference: str, h5_path, eps=1e-8, precision=8):
    summary_ref = H5Summary.load(reference)
    summary_new = H5Summary.from_h5(h5_path)

    logging.log(logging.INFO, 'Comparing Summary files ...')
    diff = summary_ref.get_diff(summary_new, eps=eps, precision=precision)
    if len(diff) != 0:
        logging.log(logging.WARN, 'H5Summary files do not match! Printing the difference:')
        print(diff)
    return len(diff) == 0

# Note: this is no longer used, but kept for reference
# def check_h5_equality(path_a, path_b):
#     # H5DIFF doesn't support excluding attributes from comparison
#     # Therefore we remove them manually
#     # Don't apply this procedure in the final version!
#     for path in [path_a, path_b]:
#         with h5py.File(path, "a") as f:
#             for attr in ['creation_date', 'file_description', 'created_by', 'Nt']:
#                 if attr in f.attrs:
#                     del f.attrs[attr]
#
#     cmd = f'h5diff -c -d 0.00000001 {path_a} {path_b}'
#     logging.log(logging.INFO, cmd)
#     res = os.system(cmd)
#     return res == 0
