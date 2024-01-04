# Examples

Many examples from k-wave-python are Python mirrors of the [large collection of great examples](http://www.k-wave.org/documentation/k-wave_examples.php) from the original k-wave project.

## Example directory layout

All examples are included as python files or ipython notebooks in a subdirectory of the example directory.
Every example has a short readme.md file which briefly describes the purpose of the example.

## List of Examples

- [Array as a sensor](at_array_as_sensor/) ([original example](http://www.k-wave.org/documentation/example_at_array_as_sensor.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/migrate-examples-to-notebooks/examples/at_array_as_sensor/at_array_as_sensor.ipynb))
- [Array as a source](array_as_source/) ([original example](http://www.k-wave.org/documentation/example_at_array_as_source.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/migrate-examples-to-notebooks/examples/at_array_as_source/at_array_as_source.ipynb))
- [Linear array transducer](at_linear_array_transducer/)
([original example](http://www.k-wave.org/documentation/example_at_linear_array_transducer.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/migrate-examples-to-notebooks/examples/at_linear_array_transducer/at_linear_array_transducer.ipynb))
    
## Contributing new examples

When adding a new example notebook, follow these steps:

1. Search the open issues to see if a related example has already been requested. If not add an example issue and assign yourself to the issue.
3. Fork and clone the repository and create a branch for your new example.
2. Create an example sub-directory using the name from the hyperlink of the origional k-wave example if it exists (e.g. for http://www.k-wave.org/documentation/example_ivp_loading_external_image.php name the directory "ivp_loading_external_image).
3. Add your example notebook or files to your example directory.
4. Add a readme.md file to your example directory briefly describing the concept or principle the example is meant to display and linking to the origonal k-wave example page if it exists.
5. Include a link in the readme.md in the examples directory to a colab notebook for your example.
6. Add a your example to the list on this readme.md and add a colab badge [using html](https://openincolab.com/) OR copy the pure markdown version above.
7. Open a pull request that [closes the open issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) from your forked example branch and name pull request "[Example] \<name of your example\>".

Thanks for contributing to k-wave-python!
