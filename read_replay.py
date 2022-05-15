import os

path = os.getcwd()
path = os.path.join(path, 'outs/replay_memory.txt')

def wreplay(memory):
    with open(path, "r+") as f:
        f.truncate()
        for i in range(len(memory)):
            f.writelines('%d: %s\n'%(i, str(memory[i])))
