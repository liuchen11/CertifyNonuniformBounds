# Certify Non-uniform Bounds

An extended version of code for paper [On Certifying Non-uniform Bounds against Adversarial Attacks](http://proceedings.mlr.press/v97/liu19h/liu19h-supp.pdf) in ICML 2019.

## Requirements

```
python       =  3.6
pytorch      >= 1.1.0
torchvision  >= 0.2.2
numpy        >= 1.16.0
matplotlib   >= 3.1.0
```

## Why we calculate certified non-uniform bounds?

In contrast to uniform bound, the non-uniform bound can certify a larger region around the data point neighborhood.
Non-uniform bound is also a proxy to study the decision boundary of neural network model and give a more fine-grained quantitation of robustness for different features.

Compared with normal models, robust models are found to have non-uniform bounds of much larger volumes and better interpretability.
The decision boundary of robust models are shown to have higher geometric similarity.

## How do we do this?

We relax the goal of maximizing the volume of certified region by the bounds of model's logits.
Then we use augumented Langrangian method to optimize the relaxed problem.
For details, please check [our paper](http://proceedings.mlr.press/v97/liu19h/liu19h-supp.pdf).

## Modules

In the repository, the main scripts to train and certify model are put under the `run` folder.
Use `python XXX.py -h` for the meaning of each hyper-parameter.

* `run/gen_syn.py`: Generate synthetic data.
* `run/train_syn.py`: Train classification models based on synthetic data.
* `run/gen_boundary.py`: Generate model's true decision boundary by brute force, only support 2-dimensional input.
* `run/train_real.py`: Train models on real dataset, support MNIST, Fashion-MNIST, SVHN.
* `run/certify.py`: Certify uniform or non-uniform bounds given a model and dataset.

## Visualization

The visualization tools are put under `plot` folder.

* `plot/syn_show.py`: Visualize the points, certified bounds and model decision boundary in the case of synthetic data.

![image](http://liuchen1993.cn/assets/Certify_Nonuniform_Bounds/github/figs/syn.png)

* `plot/real_hist.py`: Plot the histogram of bound per feature for a specific input point.

![image](http://liuchen1993.cn/assets/Certify_Nonuniform_Bounds/github/figs/real_hist.png)

* `plot/real_box`: Plot the bounding box of certified non-uniform bounds.

![image](http://liuchen1993.cn/assets/Certify_Nonuniform_Bounds/github/figs/interp.png)

## Contact

Please contact [Chen Liu](mailto:chen.liu@epfl.ch) regarding this repository.

## Citation

```
@article{liu2019certifying,
  title={On Certifying Non-uniform Bound against Adversarial Attacks},
  author={Liu, Chen and Tomioka, Ryota and Cevher, Volkan},
  journal={Thirty-sixth International Conference on Machine Learning (ICML 2019)},
  year={2019}
}
```