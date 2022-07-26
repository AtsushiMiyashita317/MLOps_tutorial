# TensorBoard tutorial

## Setup
Under appropriate virtual environment:
```sh
pip install -r requirement.txt
```

## Execute

```sh
python baseline.py
```
`Hydra`'s Multi-run
```sh
python baseline.py --multirun
```

## Show results

- vscode user

  command pallet(ctrl+shift+p)->Launch Tensorboard

- otherwise

  After this command, click the displayed url. 
  ```sh
  tensorboard --logdir ./lightning_logs
  ```
