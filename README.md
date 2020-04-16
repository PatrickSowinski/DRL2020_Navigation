# DRL2020_Navigation

[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif "Banana Env"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

![Trained Agents][image1]

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

## Table of Contents

### Training notebook

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

0. Do not use pyenv in combination with conda to manage your environment. If you want to use pyenv, you will have to set the environment to Python 3.6 there. Pyenv's global version setting will overwrite any changes you make in Anaconda.

1. Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
- __Windows__: 
```bash
conda create --name drlnd python=3.6 
activate drlnd
```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym. 
(Actually, you can ignore these extra steps:
- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
)

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies. (There is also a line in the training notebook to do this.)
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. (May not need to do this, if you set an environment with pyenv.)

![Kernel][image2]

## Was most of this README blatantly copied from Udacity's official repository, because most of the content would be basically the same anyways?

<p align="center">Yes.</p>

Video of success: https://youtu.be/QgCX-zGajT8
