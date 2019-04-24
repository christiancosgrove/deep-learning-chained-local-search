import numpy as np
import os

for file in os.listdir('./data/'):
    if '.dat' not in file and '.' in file:
        use_file = True
        full_path = './data/' + file
        print("Checking file", full_path)
        with open(full_path, 'r') as f:
            a = ""
            print(a)
            while 'NODE_COORD_SECTION' not in a:
                try:
                    a = f.readline()
                    if "EOF" in a:
                        use_file = False
                        break
                except:
                    use_file = False
                    break

            if use_file:
                print("Using file", file)
                x = []
                y = []
                while True:
                    try:
                        ln = f.readline().split()
                        x.append(float(ln[1]))
                        y.append(float(ln[2]))

                    except:
                        break

                nodes = np.zeros((len(x), 2))
                min_x = min(x)
                range_x = max(x) - min_x
                min_y = min(y)
                range_y = max(y) - min_y
                for i in range(len(x)):
                    nodes[i][0] = (x[i] - min_x) / range_x
                    nodes[i][1] = (y[i] - min_y) / range_y
                print(nodes)

                nodes.dump('./data/' + file[:-3] + 'dat')
                print("Dumped file", './data/' + file[:-3] + 'dat')
        print("Closing file", file)
        f.close()
