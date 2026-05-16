# Initial Value Problem: Photoacoustic Waveforms in 2D and 3D

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/HEAD/examples/ivp_photoacoustic_waveforms/ivp_photoacoustic_waveforms.ipynb)

See the k-wave example [here](http://www.k-wave.org/documentation/example_ivp_photoacoustic_waveforms.php), which includes one-dimensional propagation as well. 

> [!WARNING]  
> As there is no one-dimensional simulation, the example does not fully recreate the k-wave example. It underable to show how a 1D wave is different from 2D and 3D waves.

## Rational

The purpose of this example is to show that the dimensions of a system influence characteristics of the propagation. A point source in two-dimensions is represented by an infinite line source in three-dimensions. 

In two-dimensions there is cylindrical spreading, This means the acoustic energy is inversely proportional to radius, $r$, and the acoustic pressure decays as $1/\sqrt r$. In three-dimensions there is spherical spreading, so the energy is spread over $r^2$, and the pressure decays as ${1/r}$.

Photoacoustic waves in 3D have [compact support](https://en.wikipedia.org/wiki/Support_(mathematics)#Compact_support). This means they decay to zero rapidly, whereas a waveform in 2D does not have this property. As an infinite line source in 3D, there will always be some signal arriving to the detector from some (increasingly distant) part of the line source.
