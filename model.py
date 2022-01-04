#import dynet as dy

class Model():
    """Model predicts a sequence of actions, given an instruction and a starting world state.
    """
    def __init__(self):
        # TODO: implement this function. Should construct all subcomponents of the model,
        # e.g., the RNNs, or any parameters as part of the parameter collection. Should
        # initialize the optimizer.
        pass

    def _encode_seq(self, input_seq):
        """Encodes an input sequence.

        TODO: implement this function. "input_seq" is intentionally vague, and could
        include any combination of inputs: a single utterance, a sequence of utterances,
        the past utterances concatenate... it's up to you! In general, this function
        should use an RNN to encode input (natural language) tokens and return some
        hidden states (as a sequence) corresponding to the tokens.
        """
        pass

    def _encode_world_state(self, world_state):
        """Encodes an Alchemy World State.

        TODO: implement this function. Like _encode, this function can encode a single
        world state (i.e., the current world state before executing the action),
        or a sequence of world states.
        It's up to you on how to encode world states. You can use RNNs, MLPs, etc.
        """
        pass

    def _decode(self):
        """Computes probability distributions over a sequence of output tokens.

        TODO: implement this function. Consider a few things:

        1) What inputs does this function take? At minimum, it should probably take
           representations of the current utterance and the current world state. 
        2) The difference between teacher-forcing and using predicted tokens to decode.
           In the first, you feed gold tokens into the decoder at each step. This is
           mostly used during training. But during prediction, you need to feed each
           predicted token into the decoder RNN. You may want to split this function
           into two, one where you feed gold tokens and another where you feed predicted ones.

           The reason: during training, you don't need to compute an argmax (get the
           predicted tokens). You just need the probability distributions (and as dy.Expressions).
           Hint: make sure not to call .value() or any forward passes during training, and think
           about how to batch the losses per-token so the code is efficient! 

           During evaluation, you will obviously need acccess to the values of the
           probability distributions over actions in order to decode greedily (or using beam search).
        3) What's your output space look like? You could predict single-token-by-token,
           or you could collapse each action-arg1-arg2 into a single prediction. For simplicity,
           I suggest trying token-by-token first.
        """
        pass

    def train(self, batch):
        """Updates model parameters, e.g. in a batch (suggested).

        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!

        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        pass

    def predict(self, example):
        """Returns a predicted sequence given an example.

        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        pass
