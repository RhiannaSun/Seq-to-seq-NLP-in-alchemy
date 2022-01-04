from __future__ import unicode_literals, print_function, division

import warnings
warnings.filterwarnings("ignore")

import json

from argparse import ArgumentParser

from numpy.lib.function_base import average
from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import re
import random
import time

from alchemy_world_state import AlchemyWorldState
from alchemy_fsa import AlchemyFSA, NO_ARG
from one_hot_encode import one_hot_encode, one_hot_decode

from collections import Counter

from seq2seq import Seq2Seq
from lang import Lang
from encoder_decoder import Encoder, Decoder

SOS_token = 0
EOS_token = 1

UNK_thresh = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def add_padding(data_to_pad):
    max_length = -1
    for sentence in data_to_pad:
        if len(sentence.split(' ')) > max_length:
            max_length = len(sentence.split(' '))
    for i in range(len(data_to_pad)):
        pad_to_add = (max_length - len(data_to_pad[i].split(' ')))*' _PAD'
        data_to_pad[i] += pad_to_add
    return data_to_pad

def add_unk_to_train(all_instructions):
    counter = Counter([j for i in all_instructions for j in i.split(' ')])
    all_instructions_unk = []
    for i in [y if counter[y] >= UNK_thresh else '_UNK' for x in all_instructions for y in x.split(' ')]:
        if i == '_SOS':
            ins_string = '_SOS'
        elif i =='_EOS':
            ins_string += ' ' + '_EOS'
            all_instructions_unk.append(ins_string)
        else:
            ins_string += ' ' + i
    return all_instructions_unk, [i for i, value in counter.items() if value >= UNK_thresh]

def add_unk(all_instructions, vocab):
    all_instructions_unk = []
    for i in [y if y in vocab else '_UNK' for x in all_instructions for y in x.split(' ')]:
        if i == '_SOS':
            ins_string = '_SOS'
        elif i =='_EOS':
            ins_string += ' ' + '_EOS'
            all_instructions_unk.append(ins_string)
        else:
            ins_string += ' ' + i
    return all_instructions_unk

def load_data(filename, train=True):

    """Loads the data from the JSON files.

    You are welcome to create your own class storing the data in it; e.g., you
    could create AlchemyWorldStates for each example in your data and store it.

    Inputs:
        filename (str): Filename of a JSON encoded file containing the data.

    Returns:
        examples
    """
    f = open(filename)
    data = json.load(f)
    all_instructions = []
    all_prev_states = []
    all_prev_instructions = []
    all_actions = []
    all_id = []
    for i, row in enumerate(data):
        num = 0
        for j, utterance in enumerate(row['utterances']):
            id = row['identifier']
            id = id + '-' + str(num)
            num = num + 1
            all_id.append(id)

            s = utterance['instruction']
            s = re.sub(r'[^\w\s]',' ',s)
            s = re.sub(r'  ',' ',s)
            instructions = '_SOS ' + s + ' _EOS'
            if train:
                actions  = utterance['actions']
                for k in range(len(actions)):
                    if len(actions[k].split(' ')) == 2:
                        actions[k] = actions[k] + ' NONE'
                        actions[k] = actions[k].replace(' ','_')
                    else:
                        actions[k] = actions[k].replace(' ','_')
                new_actions = ['_SOS']
                new_actions.extend(actions)
                new_actions.extend(['_EOS'])
                all_actions.append(' '.join(new_actions))
            else:
                all_actions.append('_SOS')
            all_instructions.append(instructions)

            prev_state = row['utterances'][j-1]['after_env'] if j != 0 else row["initial_env"]

            all_prev_states.append(prev_state)

    all_prev_instructions = [all_instructions[i-1] if i%5!=0 else '_SOS _EOS' for i in range(len(all_instructions))]
    f.close()
    return all_id, all_instructions, all_prev_states, all_prev_instructions, all_actions


def process_data(all_instructions, all_prev_states, all_prev_instructions,all_actions):
    sentence1, sentence2, sentence3, sentence4 = all_instructions,all_prev_states, all_prev_instructions, all_actions
    source = Lang()
    target = Lang()
    pairs = []
    for i in range(len(all_instructions)):
        full = [sentence1[i],sentence2[i],sentence3[i], sentence4[i]]
        source.addSentence(sentence1[i])
        target.addSentence(sentence4[i])
        pairs.append(full)
    len_instructions = [(idx, len(instruction.split(' '))) for idx, instruction in enumerate(all_instructions)]
    seq_index_by_len= [inx for (inx, x) in sorted(len_instructions, key=lambda x: x[1])]
    pairs = [pairs[i] for i in seq_index_by_len]
    return source, target, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromWordState(world_state):
    indexes = one_hot_encode(AlchemyWorldState(world_state))
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor1 = tensorFromSentence(input_lang, pair[0])
    input_tensor2 = tensorFromWordState(pair[1])
    input_tensor3 = tensorFromSentence(input_lang, pair[2])
    target_tensor = tensorFromSentence(output_lang, pair[3])
    return (input_tensor1, input_tensor2, input_tensor3, target_tensor)



def clacModel(model, input_tensor1,input_tensor2, input_tensor3, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    # input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor1, input_tensor2, input_tensor3, target_tensor)
    num_iter = output.size(0)
    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss

def train(model, source, target, pairs, epochs=15):
    model.train()
    pad_idx = source.word2index['_PAD']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    total_loss_iterations = 0
    # training_pairs = np.array([tensorsFromPair(source, target, pair) for pair in pairs])

    batch_size = 20
    print(' ')
    print('Start training, total {} epochs'.format(epochs))
    for ep in range(epochs):
        for i in range(0, len(pairs), batch_size):
            end = min(i+batch_size, len(pairs))
            training_pair = pairs[i:end]

            training_instructions = add_padding([x for [x,y,z,a] in training_pair])
            training_prev_instructions = add_padding([z for [x,y,z,a] in training_pair])
            training_actions = add_padding([a for [x,y,z,a] in training_pair])

            training_pair = [[training_instructions[i], [y for [x,y,z,a] in training_pair][i], training_prev_instructions[i], training_actions[i]] for i in range(len(training_pair))]
            training_pair = [tensorsFromPair(source, target, pair) for pair in training_pair]

            input_tensor1 = [pair[0] for pair in training_pair]
            input_tensor2 = [pair[1] for pair in training_pair]
            input_tensor3 = [pair[2] for pair in training_pair]
            target_tensor = [pair[3] for pair in training_pair]

            input_tensor1 = torch.cat(input_tensor1[:], 1)
            input_tensor2 = torch.cat(input_tensor2[:], 1)
            input_tensor3 = torch.cat(input_tensor3[:], 1)
            target_tensor = torch.cat(target_tensor[:], 1)

            loss = clacModel(model, input_tensor1, input_tensor2, input_tensor3, target_tensor, optimizer, criterion)
            total_loss_iterations += loss

        if ep % 1 == 0:
            average_loss= total_loss_iterations / len(pairs)
            total_loss_iterations = 0
            print(f'Epoch={ep}, Loss={average_loss}, Time={time.time()}')

    torch.save(model, 'model.pt')
    print('Finished training, model saved')
    print('--------------------------------------')
    print(' ')
    return model

def evaluate(model, source, target, sentences):
    with torch.no_grad():
        input_tensor1 = tensorFromSentence(source, sentences[0])
        input_tensor2 = tensorFromWordState(sentences[1])
        input_tensor3 = tensorFromSentence(source, sentences[2])
        output_tensor = tensorFromSentence(target, sentences[3])

        decoded_words = []
        output = model(input_tensor1, input_tensor2,input_tensor3, output_tensor)
        EOS_index = target.word2index.get('_EOS')
        PAD_index = target.word2index.get('_PAD')
        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(3)
            if topi[0][0].item() == target.word2index.get('_EOS'):
                decoded_words.append('_EOS')
                # break
            elif len(decoded_words) > 1 and decoded_words[-1] == '_SOS':
                rank = 0
                while (topi[0][rank].item() == EOS_index or topi[0][rank].item() == PAD_index):
                    rank = rank + 1
                decoded_words.append(target.index2word[topi[0][rank].item()])
            else:
                decoded_words.append(target.index2word[topi[0][0].item()])

    return decoded_words

def evaluateRandomly(model, source, target, pairs, n=10):
    for i in range(1,n):
        pair = random.choice(pairs)
        print('source {}'.format(pair[0]))
        print('target {}'.format(pair[1]))
        output_words = evaluate(model, source, target, pair)
        print('origin output:', output_words)
        output_words = [x for x in output_words if len(x)>4]
        output_words = [x.replace('_',' ').replace(' NONE','') for x in output_words]
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))
        print(' ')



def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.

    TODO: This code assumes the world state is a string. However, you may sometimes
    start with an AlchemyWorldState object. I suggest loading the AlchemyWorldState objects
    into memory in load_data, and moving that part of the code to load_data. The following
    code just serves as an example of how to 1) make an AlchemyWorldState and 2) execute
    a sequence of actions on it.

    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)
    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]

        # JSON file doesn't contain  NO_ARG.
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]
        fsa.feed_complete_action(act, arg1, arg2)

    return fsa.world_state()

def predict(model, source, target, all_id, all_prev_states, data, outname, mode):
    """Makes predictions for data given a saved model.

    This function should predict actions (and call the AlchemyFSA to execute them),
    and save the resulting world states in the CSV files (same format as *.csv).

    TODO: you should implement both "gold-previous" and "entire-interaction"
        prediction.

    In the first case ("gold-previous"), for each utterance, you start in the previous gold world state,
    rather than the on you would have predicted (if the utterance is not the first one).
    This is useful for analyzing errors without the problem of error propagation,
    and you should do this first during development on the dev data.

    In the second case ("entire-interaction"), you are continually updating
    a single world state throughout the interaction; i.e. for each instruction, you start
    in whatever previous world state you ended up in with your prediction. This method can be
    impacted by cascading errors -- if you made a previous incorrect prediction, it's hard
    to recover (or your instruction might not make sense).

    For test labels, you are expected to predict /final/ world states at the end of each
    interaction using "entire-interaction" prediction (you aren't provided the gold
    intermediate world states for the test data).

    Inputs:
        model (Model): A seq2seq model for this task.
        data (list of examples): The data you wish to predict for.
        outname (str): The filename to save the predictions to.
    """
    if mode == 'instruction':
        outputs = []
        actions = []
        prev_states = []
        instructions = []
        for i in range(len(data)):
            pair = data[i]
            output_words = evaluate(model, source, target, pair)
            # print out result for viewing
            prev_states.append(all_prev_states[i])
            instructions.append(pair[0])
            actions.append(output_words)

            output_words = [x for x in output_words if len(x)>4]
            action_series = [x.replace('_',' ').replace(' NONE','') for x in output_words]
            output = execute(all_prev_states[i], action_series)
            if output == None:
                if len(outputs) == 0:
                    outputs.append('1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_')
                else:
                    outputs.append(outputs[-1])
            else:
                output = output.components()
                output_str = ""
                for i in range(len(output)):
                    output_str += str(i+1) + ':'
                    if len(output[i]) == 0:
                        output_str += '_ '
                    else:
                        for j in range(len(output[i])):
                            output_str += (output[i][j])
                        output_str += ' '
                outputs.append(output_str[:-1])
        output_instruction = pd.DataFrame(columns=['id','final_world_state'])
        output_instruction['id'] = all_id
        output_instruction['final_world_state'] = outputs

        output_action = pd.DataFrame(columns=['id', 'prev_state', 'instructions', 'action'])
        output_action['id'] = all_id
        output_action['prev_state'] = prev_states
        output_action['instructions'] = instructions
        output_action['action'] = actions
        output_action.to_csv(outname + '_action_pred.csv',index=False)


        output_interaction = output_instruction[output_instruction['id'].str[-1] == '4']
        output_interaction['id'] = output_interaction['id'].apply(lambda x : x[:-2])

        output_instruction.to_csv(outname + '_instruction_pred.csv',index=False)

    else:
        if outname == 'dev':
            all_prev_states = [all_prev_states[i] if i%5 == 0 else '' for i in range(len(all_prev_states))]
        outputs = []
        actions = []
        instructions = []
        for i in range(len(data)):
            pair = data[i]
            if all_prev_states[i] == '':
                all_prev_states[i] = outputs[-1]
            pair[1] = all_prev_states[i]
            output_words = evaluate(model, source, target, pair)

            instructions.append(pair[0])
            actions.append(output_words)

            output_words = [x for x in output_words if len(x)>4]
            action_series = [x.replace('_',' ').replace(' NONE','') for x in output_words]
            output = execute(all_prev_states[i], action_series)
            if output == None:
                if len(outputs) == 0:
                    state_output = all_prev_states[0]
                else:
                    state_output = outputs[-1]
            else:
                output = output.components()
                output_str = ""
                mistake = False
                for i in range(len(output)):
                    output_str += str(i+1) + ':'
                    if len(output[i]) == 0:
                        output_str += '_ '
                    else:
                        if len(output[i]) <= 4:
                            for j in range(len(output[i])):
                                output_str += (output[i][j])
                        else:
                            mistake = True
                        output_str += ' '
                if mistake:
                    state_output = pair[1]
                else:
                    state_output = output_str[:-1]
            outputs.append(state_output)
        output_instruction = pd.DataFrame(columns=['id','final_world_state'])
        output_instruction['id'] = all_id
        output_instruction['final_world_state'] = outputs
        output_instruction.to_csv(outname + '_instruction_pred.csv',index=False)

        output_action = pd.DataFrame(columns=['id', 'prev_state', 'instructions', 'action'])
        output_action['id'] = all_id
        output_action['prev_state'] = all_prev_states
        output_action['instructions'] = instructions
        output_action['action'] = actions
        output_action.to_csv(outname + '_action_pred.csv',index=False)

        output_interaction = output_instruction[output_instruction['id'].str[-1] == '4']
        output_interaction['id'] = output_interaction['id'].apply(lambda x : x[:-2])

        output_interaction.to_csv(outname + '_interaction_pred.csv',index=False)



def main():
    # A few command line arguments
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--predict", type=bool, default=False)
    parser.add_argument("--mode", default='instruction')
    parser.add_argument("--saved_model", type=str, default="")
    args = parser.parse_args()

    assert args.train or args.predict

    # Load the data; you can also use this to construct vocabularies, etc.
    # train_data = load_data("train.json")
    # dev_data = load_data("dev.json")

    # Construct a model object.
    model = Model()

    if args.train:
        # Trains the model
        all_id, all_instructions, all_prev_states, all_prev_instructions, all_actions = load_data("train.json")

        # add unk
        all_instructions, vocab = add_unk_to_train(all_instructions)
        all_prev_instructions = add_unk(all_prev_instructions,vocab)
        source, target, pairs = process_data(all_instructions, all_prev_states, all_prev_instructions, all_actions)
        randomize = random.choice(pairs)
        print(' ')
        print('-------------Sample-------------------')
        print('random input {}'.format(randomize[:-1]))
        print('random output {}'.format(randomize[-1]))
        print('--------------------------------------')
        print(' ')
        #print number of words
        input_size = source.n_words
        output_size = target.n_words

        # print('Input : {} Output : {}'.format(input_size, output_size))

        epochs = 1
        enc_dropout = 0.05
        dec_dropout = 0.05

        #create instruction encoder
        embed_size = 50
        hidden_size = 100
        num_layers = 1

        encoder_instruction = Encoder(input_size, hidden_size, embed_size, num_layers, enc_dropout)

        # create 7 beaker encoder
        input_size_beaker = 4 * 6 + 1
        hidden_size_beaker = 20
        embed_size = 50
        encoders_world = [Encoder(input_size_beaker, hidden_size_beaker, embed_size, num_layers, enc_dropout) for i in range(7)]

        decoder = Decoder(output_size, 2*hidden_size+7*hidden_size_beaker, embed_size, num_layers, dec_dropout)

        model = Seq2Seq(encoder_instruction,encoders_world, decoder, device).to(device)

        #print model
        # print(encoder1)
        # print(encoder2)
        # print(decoder)
        model = train(model, source, target, pairs, epochs)
        # evaluateRandomly(model, source, target, pairs)

    if args.predict:
        if not args.train:
            assert args.saved_model
            model = torch.load(args.saved_model)

            # Get source and target lang models
            all_id, all_instructions, all_prev_states, all_prev_instructions, all_actions  = load_data("train.json")

            # add unk
            all_instructions, vocab = add_unk_to_train(all_instructions)
            all_instructions = add_padding(all_instructions)
            all_actions = add_padding(all_actions)
            source, target, _ = process_data(all_instructions, all_prev_states, all_prev_instructions,all_actions)
        # Makes predictions for the data, saving it in the CSV format
        # assert args.saved_model

        # TODO: you can modify this to take in a specified split of the data,
        # rather than just the dev data.

        data_names = [('dev', "dev.json"), ('test_leaderboard', 'test_leaderboard.json')]
        for data_type, file_name in data_names:
            print(' ')
            print('--------------------------------------')
            print(f'Read in {data_type} data')
            all_id, all_instructions, all_prev_states, all_prev_instructions, all_actions = load_data(file_name, train=False)
            all_instructions = add_unk(all_instructions,vocab)
            all_prev_instructions = add_unk(all_prev_instructions,vocab)
            pairs = []
            for i in range(len(all_instructions)):
                full = [all_instructions[i], all_prev_states[i], all_prev_instructions[i], all_actions[i]]
                pairs.append(full)
            print('Start predicting...')
            predict(model, source, target, all_id, all_prev_states, pairs, data_type, args.mode)
            print('Result saved')
            print('--------------------------------------')
            print(' ')

if __name__ == "__main__":
    main()
