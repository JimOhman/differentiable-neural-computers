# Differentiable Neural Computers
 A Pytorch implementation of the [Differentiable Neural Computer](https://deepmind.com/blog/article/differentiable-neural-computers).

* [x] Improvements added from [paper A](https://home.ttic.edu/~klivescu/MLSLP2017/MLSLP2017_ben-ari.pdf) and [paper B](https://arxiv.org/pdf/1904.10278.pdf).
* [x] Training results are logged with [Tensorboard](https://github.com/tensorflow/tensorboard).
* [x] Evaluate and visualize saved models.
* [x] Repeat-Copy task
* [ ] bAbI task
 
 ## Repeat-Copy Example:
 
 <img src="https://user-images.githubusercontent.com/49884398/137722981-4c09f67e-2f3d-4524-85ac-3fb35fc1bb0a.gif" width="450" height="320"/> <img src="https://user-images.githubusercontent.com/49884398/137723640-f451e6f1-a5fc-4b53-afd3-43802852cc7e.png" width="250" height="320"/>

 ## Installation:
```bash
git clone https://github.com/JimOhman/differentiable-neural-computers.git
cd differentiable-neural-computers
pip install -r requirements.txt
```

## Reproduce example:

```bash
python train.py --data_seed 12 --seed 12 --autoclip --batch_size 256 --pattern_width 9 --max_repeats 2 --num_patterns 3 --input_dim 9 --output_dim 9 --use_mask --free_strengths
```

See training results with tensorboard:
```bash
tensorboard --logdir runs
```

Visualize saved models:
```bash
python test.py --saves_dir runs/.../saves --visualize --seed 6
```
