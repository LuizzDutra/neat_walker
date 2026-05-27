from pathlib import Path
import pickle
from train_models import create_run_net, get_config

cwd = Path.cwd()
winners = cwd / "winners"

winners_list = list(winners.glob("*.pkl"))

for i, w in enumerate(winners_list):
    print(i, w.name)

choice_n = int(input("Choose which to run\n"))

model_path = winners_list[choice_n]


with open(model_path, "rb") as f:
    net = pickle.load(f)

create_run_net(net, get_config())
