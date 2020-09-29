
filtered_arguments = ["save_path", "seed", "dataset_path", "mode"]

def __convert_string(arg):
    if arg is True:
        return "True"
    elif arg is False:
        return "False"
    elif isinstance(arg, int) or isinstance(arg, float):
        return str(arg)
    return arg

# performance.type: pd.DataFrame
def write_performance(configs, performance, path):
    with open(path, "a") as f:
        all_arguments = sorted(filter(lambda x: x[0] not in filtered_arguments, list(vars(configs).items())))
        for key, value in all_arguments:
            f.write("%-8s" % key + ":" + "%-5s" % value + "|")
        f.write("\n")
        f.write(performance.to_string())
