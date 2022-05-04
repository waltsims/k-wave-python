import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='k-Wave-python',
                 version='0.1.0',
                 description='Acoustics toolbox for time domain acoustic '
                             'and ultrasound simulations in complex and tissue-realistic media.',
                 author='Farid Yagubbayli',
                 author_email='farid.yagubbayli@tum.de',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 maintainer='Walter Simson',
                 maintainer_email='walter.simson@tum.de',
                 url='https://github.com/waltsims/k-wave-python',
                 project_urls={'Original Project Page': 'http://www.k-wave.org/',
                               'Github Repo': 'https://github.com/waltsims/k-wave-python',
                               'Documentation': 'https://waltersimson.com/k-wave-python/'},
                 packages=['kwave', 'kwave.utils', 'kwave.reconstruction', 'kwave.kWaveSimulation_helper'],
                 package_data={'kwave': ['bin/*']},
                 include_package_data=True,
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires="~=3.8",
                 )
