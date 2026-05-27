from pathlib import Path
from src.simulation.simulate import create_run_net
from src.training.configs import get_config
from src.results.manager import get_net, get_saved_nets

cwd = Path.cwd()
winners = cwd.parent / "winners"

saved_nets = get_saved_nets()
for i, n in enumerate(saved_nets):
    print(i, n.name)

if len(saved_nets) > 0:
    choice_n = int(input("Choose which to run\n"))

    net = get_net(saved_nets[choice_n].name)

    create_run_net(net, get_config())
else:
    print("No results saved")
