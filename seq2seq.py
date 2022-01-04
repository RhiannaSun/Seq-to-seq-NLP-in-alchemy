import torch
import torch.nn as nn
import random
import numpy as np

SOS_token = 0
EOS_token = 1
class Seq2Seq(nn.Module):
    def __init__(self, encoder1, encoders_world, decoder, device):
        super().__init__()

        #initialize the encoder and decoder
        self.encoder1 = encoder1
        self.encoders_world = encoders_world
        self.decoder = decoder
        self.device = device

    def forward(self, source1, source2, source3, target, teacher_forcing_ratio=0.5):

        # input_length = source1.size(0) #get the input length (number of words in sentence)

        batch_size = target.size(1)
        target_length = 8 if batch_size == 1 else target.size(0)
        vocab_size = self.decoder.output_dim

        #initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)
        hidden, cell = self.encoder1(source1)
        for i in range(7):

            hidden1, cell1 = self.encoders_world[i](source2[i*4:i*4+4])
            hidden = torch.cat((hidden,hidden1),2)
            cell = torch.cat((cell,cell1),2)

        hidden1, cell1 = self.encoder1(source3)
        hidden = torch.cat((hidden,hidden1),2)
        cell = torch.cat((cell,cell1),2)
        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]
        for t in range(1, target_length):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x,hidden, cell)
            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            if batch_size > 1:
                # if training
                x = target[t] if random.random() < teacher_forcing_ratio else best_guess
            else:
                # if predict, take last output
                x = best_guess

        return outputs
    teacher_forcing_ratio = 0.5