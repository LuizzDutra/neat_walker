from pathlib import Path
import pickle

root = Path.cwd()
winners_path = root / "winners"

def save_net(obj, name: str):
   winners_path.mkdir(exist_ok=True) 
   with open(winners_path / name, "wb") as f:
       pickle.dump(obj, f)

def get_net(name: str):
    winners_path.mkdir(exist_ok=True) 
    with open(winners_path / name, "rb") as f:
        net = pickle.load(f)
    return net

def get_saved_nets():
    winners_list = list(winners_path.glob("*.pkl"))
    return winners_list

