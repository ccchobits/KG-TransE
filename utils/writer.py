import time

filtered_arguments = ["save_path", "seed", "dataset_path", "mode", "log"]
span = {
    "bern": 5,
    "bs": 5,
    "dataset": 5,
    "dim": 3,
    "epoch": 4,
    "init_lr": 4,
    "lr_decay": 3,
    "margin": 3,
    "norm": 1,
    "model": 10
}
order = {
    "model": 1,
    "bern": 2,
    "bs": 3,
    "dataset": 4,
    "dim": 5,
    "epoch": 6,
    "init_lr": 7,
    "lr_decay": 8,
    "margin": 9,
    "norm": 10,
}
# performance.type: pd.DataFrame
def write_performance(configs, performance, path):
    with open(path, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "|")
        all_arguments = sorted(filter(lambda x: x[0] not in filtered_arguments, list(vars(configs).items())),
                               key=lambda x: order[x[0]])
        for key, value in all_arguments:
            f.write(key + ":" + ("%-"+ str(span[key]) +"s") % value + "|")
        f.write("\n")
        f.write(performance.to_string() + "\n")
