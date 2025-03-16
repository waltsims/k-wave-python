# Using An Ultrasound Transducer As A Sensor Example

<a target="_blank" href="https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/na_controlling_the_pml/na_controlling_the_pml.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Running the example

Please have a look at the `example_na_controlling_the_pml.ipynb` notebook file to see the example in action.

## Preparing the Matlab reference files

We can verify the correctness of Python implementation by saving intermediate/final variable states from the Matlab script
and comparing them against Python version. Saving the variable states can be done manually by running the Matlab script,
pausing it in appropriate places and saving the variables using `save(...)` method. However, this is very involved process.
To automate the process a bit, we followed the steps below.

1. Grab the original example file from K-wave repository (link: [example_na_controlling_the_PML.m](https://github.com/ucl-bug/k-wave/blob/main/k-Wave/examples/example_na_controlling_the_PML.m))
2. Modify the file such that we are running for all scenarios (4 different simulation cases) and saving the output sensor data + visualized figures each time. We saved the modified script in the `modified_matlab_example.m` file.
3. Run the modified file from the step above. Now we should have 4x `.mat` and `.png` files for the captured sensor data and visualized figure, respectively.
4. We check against these files in the `example_na_controlling_the_pml.ipynb` notebook.
