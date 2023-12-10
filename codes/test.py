import json
import subprocess
import shlex

lens = [1000, 5000, 10000, 100000, 1000000, 2000000]
inits = ["N", "O"]
q_sizes = range(0, 6)


def run_program(name, parameters):
    command = f"./{name} {parameters}"
    result = subprocess.run(
        shlex.split(command), capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def grid_search(program, lens, inits, sizes):
    time_list = []
    for data_len in lens:
        for init in inits:
            for size in sizes:
                parameters = f"{data_len} {init} {size}"
                time = 0
                for i in range(5):
                    output = run_program(program, parameters)
                    time += float(output.split(":")[1])
                item = {
                    "data_len": data_len,
                    "init": init,
                    "size": size,
                    "time": time / 5,
                }
                time_list.append(item)
    of = open("grid_search_result.json", "w")
    of.write(json.dumps(time_list, indent=2))


if __name__ == "__main__":
    grid_search("main", lens, inits, q_sizes)
