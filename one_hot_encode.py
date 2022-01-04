from alchemy_world_state import AlchemyWorldState
import numpy as np
COLORS = ['y', 'o', 'r', 'g', 'b', 'p']

def one_hot_encode(aws):
    encode = np.zeros([7,4,6])
    encode_index = []
    # env = AlchemyWorldState(all_prev_states[0]).components()
    env = aws.components()
    for i in range(len(env)):
        beaker = env[i]
        num_in_beaker = len(beaker)
        if num_in_beaker != 0:

            for j in range(num_in_beaker):
                # print(beaker[j])
                color_i = COLORS.index(beaker[j])
                encode[i][j][color_i] = 1
        flat_code = encode[i].reshape(1,4*6)[0]
        index = [i  for i, value in enumerate(flat_code) if value == 1]
        while len(index) < 4:
            index.append(24)
        encode_index.append(index)
    return encode_index

def one_hot_decode(encoded_env):
    full_decode = ''
    for i in range(7):
        decode = ''
        decode += (str(i+1) + ':')
        count = 0
        color = None
        for j in range(i *24, (i+1) *24):
            if encoded_env[j] != 0:
                count = count + 1
                color = (j % 24) % 6
        # print(color)
        if count == 0:
            decode += ('_ ')
        else:
            decode += COLORS[color] * count
        full_decode =full_decode + decode + ' '
    return full_decode
