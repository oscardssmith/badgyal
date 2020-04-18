import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys
import os
import numpy as np
from collections import defaultdict


CACHE=100000
MAX_BATCH = 8
MIN_POLICY=0.2
WDL = np.array([-1., 0., 1.])


class AbstractNet:
    def __init__(self, cuda=True):
        self.net = self.load_net()
        self.cuda = cuda
        if self.cuda:
            self.net = self.net.cuda()
        self.net.eval()
        self.cache = pylru.lrucache(CACHE)
        self.prefetch = {}

    def process_boards(self, boards):
        input = bulk_board2planes(boards)
        if self.cuda:
            input = input.pin_memory().cuda(non_blocking = True)
        with torch.no_grad():
            policies, values = self.net(input)
            return policies.cpu(), values.cpu()

    def cache_boards(self, boards, softmax_temp=1.61):
        for b in boards:
            epd = b.epd()
            if not epd in self.cache:
                self.prefetch[epd] = b

        if len(self.prefetch) > MAX_BATCH:
            policies, values = self.process_boards(self.prefetch.values())
            with torch.no_grad():
                for i, b in enumerate(self.prefetch.values()):
                    inp = policies[i].unsqueeze(dim=0)
                    policy = policy2moves(b, inp, softmax_temp=softmax_temp)
                    value = values[i]
                    value = self.value_to_scalar(value)
                    self.cache[b.epd()] = [policy, value]
            self.prefetch = {}

    def cache_eval(self, board):
        epd = board.epd()
        if epd in self.cache:
            return self.cache[epd]
        else:
            return None, None

    def value_to_scalar(self, value):
        if not self.classical:
            return np.dot(WDL, value)
        return value.item()

    def eval(self, board, softmax_temp=1.61):
        epd = board.epd()
        if epd in self.cache:
            policy, value = self.cache[epd]
        else:
            # put all the child positions on the board
            boards = [board.copy()]
            policies, values = self.process_boards(boards)


            with torch.no_grad():

                for i, b in enumerate(boards):
                    inp = policies[i].unsqueeze(dim=0)
                    policy = policy2moves(b, inp, softmax_temp=softmax_temp)
                    value = values[i]
                    value = self.value_to_scalar(value)
                    self.cache[b.epd()] = [policy, value]

            policy, value = self.cache[epd]

        # get the best move and prefetch it
        tocache = []

        for m, val in policy.items():
            if val >= MIN_POLICY:
                bd = board.copy()
                bd.push_uci(m)
                tocache.append(bd)
        if (len(tocache) < 1):
            m = max(policy, key = lambda k: policy[k])
            bd = board.copy()
            bd.push_uci(m)
            tocache.append(bd)
        self.cache_boards(tocache, softmax_temp=softmax_temp)

        # return the values
        return policy, value

    def bulk_eval(self, boards, softmax_temp=1.61):

        retval_p = []
        retval_v = []

        policies, values = self.process_boards(boards)

        with torch.no_grad():
            for i, b in enumerate(boards):
                inp = policies[i].unsqueeze(dim=0)
                policy = policy2moves(b, inp, softmax_temp=softmax_temp)
                value = values[i]
                value = self.value_to_scalar(value)
                retval_p.append(policy)
                retval_v.append(value)
                self.cache[b.epd()] = [policy, value]

        return retval_p, retval_v

class LoadedNet(AbstractNet):
    def __init__(self, path, channels=128, blocks=10, se=4, policy_channels=None, classical=True, cuda=True):
        self.path = path
        self.channels = channels
        self.blocks = blocks
        self.se = se
        if policy_channels == None:
            self.policy_channels = channels
        else:
            self.policy_channels = policy_channels
        self.classical = classical
        super().__init__(cuda=cuda)
        
    def load_net(self):
        cwd = os.path.abspath(os.path.dirname(__file__))
        full_path = os.path.join(cwd, self.path)
        net = model.Net(self.channels,
                        self.blocks,
                        self.policy_channels,
                        self.se,
                        classical=self.classical)
        if self.classical:
            net.import_proto_classical(full_path)
        else:
            net.import_proto(full_path)
        return net

class MultiNet(AbstractNet):
    def __init__(self, nets):
        self.nets = nets
    
    def __call__(self, cuda=True):
        self.nets = [net(cuda=cuda) for net in self.nets]
        return self
        
    def eval(self, board, softmax_temp=1.61):
        num_nets = len(self.nets)
        policy_avg = defaultdict(float)
        value_tot = 0
        for net in self.nets:
            policy, value = net.eval(board, softmax_temp)
            value_tot += value
            for move, p in policy.items():
                policy_avg[move] += p
        for move in policy_avg.keys():
            policy_avg[move] /= num_nets
        return policy_avg, value_tot/num_nets
            

    def bulk_eval(self, boards, softmax_temp=1.61):
        num_nets = len(self.nets)
        policy_avg = defaultdict(float)
        value_tot = 0
        for net in self.nets:
            policies, values = net.bulk_eval(boards, softmax_temp)
            print(policies, values)
