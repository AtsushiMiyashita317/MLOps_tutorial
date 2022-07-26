# MLFlow tutorial

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

## Show result

- vscode user

  command pallet(ctrl+shift+p)->Launch Tensorboard

- 非vscodeユーザー

  After this command, click the displayed url
  ```sh
  tensorboard --logdir ./lightning_logs
  ```
