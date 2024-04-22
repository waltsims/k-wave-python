# Examples

Many examples from k-wave-python are Python mirrors of the [large collection of great examples](http://www.k-wave.org/documentation/k-wave_examples.php) from the original k-wave project.
All examples are included as python files or notebooks in a subdirectory of the example directory.
Every example has a short readme.md file which briefly describes the purpose of the example.

## List of Examples

- [Array as a sensor](at_array_as_sensor/) ([original example](http://www.k-wave.org/documentation/example_at_array_as_sensor.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_array_as_sensor/at_array_as_sensor.ipynb))
- [Array as a source](at_array_as_source/) ([original example](http://www.k-wave.org/documentation/example_at_array_as_source.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_array_as_source/at_array_as_source.ipynb))
- [Linear array transducer](at_linear_array_transducer/)
([original example](http://www.k-wave.org/documentation/example_at_linear_array_transducer.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_linear_array_transducer/at_linear_array_transducer.ipynb))
- [Photoacoustic Waveforms](ivp_photoacoustic_waveforms/) ([original example](http://www.k-wave.org/documentation/example_ivp_photoacoustic_waveforms.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/ivp_photoacoustic_waveforms/ivp_photoacoustic_waveforms.ipynb))
- [Controlling the PML](na_controlling_the_pml/)
([original example](http://www.k-wave.org/documentation/example_na_controlling_the_pml.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/na_controlling_the_pml/na_controlling_the_pml.ipynb))
- [Linear transducer B-mode](us_bmode_linear_transducer/) ([original example](http://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/us_bmode_linear_transducer/us_bmode_linear_transducer.ipynb))    
- [Phased array B-mode](us_bmode_phased_array/)
([original example](http://www.k-wave.org/documentation/example_us_bmode_phased_array.php), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/us_bmode_phased_array/us_bmode_phased_array.ipynb))
- [Circular piston transducer](at_circular_piston_3D/) ([original example](http://www.k-wave.org/documentation/example_at_piston_and_bowl_transducers.php#heading3), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_circular_piston_3D/at_circular_piston_3D.ipynb))

- [Circular piston transducer (axisymmetric)](at_circular_piston_AS/) ([original example](http://www.k-wave.org/documentation/example_at_piston_and_bowl_transducers.php#heading4), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_circular_piston_AS/at_circular_piston_AS.ipynb))

- [Focused bowl transducer](at_focused_bowl_3D/) ([original example](http://www.k-wave.org/documentation/example_at_piston_and_bowl_transducers.php#heading5), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_focused_bowl_3D/at_focused_bowl_3D.ipynb))

- [Focused bowl transducer (axisymmetric)](at_focused_bowl_AS/) ([original example](http://www.k-wave.org/documentation/example_at_piston_and_bowl_transducers.php#heading6), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_focused_bowl_AS/at_focused_bowl_AS.ipynb))



- [Focused annular array](at_focused_annular_array_3D/) ([original example](http://www.k-wave.org/documentation/example_at_piston_and_bowl_transducers.php#heading7), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/at_focused_annular_array_3D/at_focused_annular_array_3D.ipynb))


 


## Contributing new examples

When adding a new example notebook, follow these steps:

1. Search the open issues to see if a related example has already been requested. If not add an example issue and assign yourself to the issue.
3. Fork and clone the repository and create a branch for your new example.
2. Create an example sub-directory using the name from the hyperlink of the original k-wave example if it exists (e.g. for http://www.k-wave.org/documentation/example_ivp_loading_external_image.php name the directory "ivp_loading_external_image).
3. Add your example notebook to your example directory.
4. Create a Python script that mirrors your example notebook in the same directory.
5. Add a README.md file to your example directory briefly describing the concept or principle the example is meant to display and linking to the origonal k-wave example page if it exists.
6. Include a link in the readme.md in the examples directory to a colab notebook for your example.
7. Add a your example to the list on this readme.md and add a colab badge [using html](https://openincolab.com/) OR copy the pure markdown version above.
8. Open a pull request that [closes the open issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) from your forked example branch and name pull request "[Example] \<name of your example\>".

Thanks for contributing to k-wave-python!
