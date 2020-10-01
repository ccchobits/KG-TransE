import time

# log: .type: pandas.DataFrame
def write_log(log):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    path =  "../scripts/asset/log/" + current_time + ".log"
    with open(path, "w") as f:
        f.write(log.to_string())
