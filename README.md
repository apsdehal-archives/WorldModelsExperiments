# World Models Experiments

## Generating expert trajectories for Ke et.al. 2019 (ICLR'19)

Inside carracing folder run:

```
python generation_script.py --threads 10
```

This will generate expert rollout trajectories in `expert_rollouts` folder which can be used for CarRacing experiments in the main repo. On running this script, 10 threads will be launched, each of which will generate 1000 expert rollouts leading to 10000 expert rollouts. Change number of threads and other parameters accordingly if you have more CPUs.

You can change the following numbers to x / number of threads where x is total number of rollouts you want.
- [carracing/generation_script.py#L16](carracing/generation_script.py#L16)
- [carracing/model.py#L279](carracing/model.py#L279)

Step by step instructions of reproducing [World Models](https://worldmodels.github.io/) ([pdf](https://arxiv.org/abs/1803.10122)).

![World Models](https://worldmodels.github.io/assets/world_models_card_both.png)

Please see [blog post](http://blog.otoro.net//2018/06/09/world-models-experiments/) for step-by-step instructions.

# Note regarding OpenAI Gym Version

Please note the library versions in the blog post. In particular, the experiments work on gym 0.9.x and does NOT work on gym 0.10.x. You can install the older version of gym using the command `pip install gym==0.9.4`, `pip install numpy==1.13.3` etc.

# Citation

If you find this project useful in an academic setting, please cite:

```
@article{Ha2018WorldModels,
  author = {Ha, D. and Schmidhuber, J.},
  title  = {World Models},
  eprint = {arXiv:1803.10122},
  doi    = {10.5281/zenodo.1207631},
  url    = {https://worldmodels.github.io},
  year   = {2018}
}
```

# Issues

For general discussion about the World Model article, there are already some good discussion threads here in the GitHub [issues](https://github.com/worldmodels/worldmodels.github.io/issues) page of the interactive article. Please raise issues about this specific implementation in the [issues](https://github.com/hardmaru/WorldModelsExperiments/issues) page of this repo.

# Licence

MIT
