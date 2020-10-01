import time

filtered_arguments = ["save_path", "seed", "dataset_path", "mode"]

# performance.type: pd.DataFrame
def write_performance(configs, performance, path):
    print("bern2: %s" % str(configs.bern), flush = True)
    with open(path, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "|")
        all_arguments = sorted(filter(lambda x: x[0] not in filtered_arguments, list(vars(configs).items())))
        for key, value in all_arguments:
            f.write(key + ":" + "%-5s" % value + "|")
        f.write("\n")
        f.write(performance.to_string() + "\n")
