import platform
from skimage.draw import line
from prettytable import PrettyTable
import os

OPSYS = platform.system()
SEP = '\\' if OPSYS == 'Windows' else '/'
PREFIX = '/mnt/c' if OPSYS == 'Linux' else 'C:'

# Check intermediate layers and sizes
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    table.reversesort = True
    layers = model.named_parameters()
    for name, parameter in layers:
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table.get_string(sortby="Parameters"))
    # print(table)
    table.sortby = "Parameters"
    print(f"Total Trainable Params: {total_params}")
    return total_params


def linemaker(p_start, p_end, thickness=1):
    x_start, x_end, y_start, y_end = p_start[0], p_end[0], p_start[1], p_end[1]
    
    x_diff, y_diff = x_end - x_start, y_end -y_start
    m = y_diff / x_diff if x_diff != 0 else 10.

    lines = []
    or_line = [coord for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]
    lines += [[coord] for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]

    line_factors = []
    for i in range(thickness-1):
        sign = 1
        if i%2 != 0:
            sign = -1
        i = int(i/2.)+1
        # th=2: +1, th=3: [+1,-1], th=4: [+1,-1,+2]
        line_factors.append(sign*i)

    for factor in line_factors:

        if abs(m) > 1:
            extra_line = list(zip(*line(*(p_start[0]+factor, p_start[1]), *(p_end[0]+factor, p_end[1]))))
            # extra_line = list(zip(*_line_profile_coordinates((x_start+1, y_start), (x_end+1, y_end), linewidth=1)))
        else:
            extra_line = list(zip(*line(*(p_start[0], p_start[1]+factor), *(p_end[0], p_end[1]+factor))))
            # extra_line = list(zip(*_line_profile_coordinates((x_start, y_start+1), (x_end, y_end+1), linewidth=1)))
        # lines += extra_line
        for idx in range(len(lines)):
            lines[idx].append(extra_line[idx])

        # check if all points are offsetted correctly
        for c_line_or, c_line_ex in zip(or_line, extra_line):
            if sum(c_line_ex) != sum(c_line_or)+factor:
                hi = 1

    return lines


class TQDMBytesReader(object):
    """ from https://stackoverflow.com/questions/30611840/pickle-dump-with-progress-bar """
    def __init__(self, fd, **kwargs):
        self.fd = fd
        from tqdm import tqdm
        self.tqdm = tqdm(**kwargs)

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)


def dir_maker(store_folder_path, description_log):
    if os.path.isdir(store_folder_path):
        print('Path already exists!')
        quit()
    else:
        os.mkdir(store_folder_path)
        with open(os.path.join(store_folder_path, 'description.txt'), 'w') as f:
            f.write(description_log)
        f.close()