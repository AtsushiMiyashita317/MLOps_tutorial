# TensorBoard tutorial

## setup
Under appropriate virtual environment:
```sh
pip install -r requirement.txt
```

## execute

```sh
python baseline.py
```
`Hydra`'s Multi-run
```sh
python baseline.py --multirun
```

## show results

- vscode user

  command pallet(ctrl+shift+p)->Launch Tensorboard

- otherwise

  After this command, click the displayed url. 
  ```sh
  tensorboard --logdir ./lightning_logs
  ```
