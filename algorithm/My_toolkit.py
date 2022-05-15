# coding=utf-8
import os
import numpy as np

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def mkdir( rank, experiment_name):
    """传进来一个实验名， 生成 ./logs ./result ./model 生成对应文件夹"""
    folder = os.path.exists("logs/"+experiment_name )

    if not folder:
        os.makedirs("logs/" + experiment_name)
    if not os.path.exists("logs/"+experiment_name + "/DDPG_" + str(rank)):
        os.makedirs("logs/" + experiment_name + "/DDPG_" + str(rank))

    folder = os.path.exists("result/" + experiment_name)
    if not folder:
        os.makedirs("result/" + experiment_name)

    folder = os.path.exists("model/" + experiment_name)
    if not folder:
        os.makedirs("model/" + experiment_name)

def wreplay(memory):
    path = os.getcwd()
    path = os.path.join(path, 'outs/replay_memory.txt')
    with open(path, "r+") as f:
        f.truncate()
        f.write('shape:\nobs:%s\nsta:%s\naction:%s\nreward:%s\nobs1:%s\nsta1:%s\nter:%s\n'%
        (str(memory[0].shape), str(memory[1].shape), str(memory[2].shape), str(memory[3].shape),
        str(memory[4].shape), str(memory[5].shape), str(memory[6].shape)))
        f.write('obs: %s\n'%str(memory[0]))
        f.write('sta: %s\n'%str(memory[1]))
        f.write('action: %s\n'%str(memory[2]))
        f.write('reward: %s\n'%str(memory[3]))
        f.write('obs1: %s\n'%str(memory[4]))
        f.write('sta1: %s\n'%str(memory[5]))
        f.write('ter: %s\n'%str(memory[6]))

    # np.savez(path, *memory)

# if __name__ == '__main__':
    # path = os.getcwd()
    # path = os.path.join(path, 'outs/replay_memory.txt')

    # a = np.zeros((3,3,3)).astype('float32')
    # b = np.ones((2,3)).astype('float32')
    # c = np.array([1,2])
    # d = [a, b, c]

    # with open(path, "r+") as f:
    #     for i in range(len(d)):
    #         f.write(str(d[i]))

    # # n = np.load(path)
    # # print(n.files)
    # # print(n["arr_0"])
    # # print(n["arr_1"])
