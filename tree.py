import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, dim=-1)
        return F.one_hot(idx, num_classes=input.shape[-1]).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SoftDecisionTree(nn.Module):
    def __init__(self, num_features, height, num_classes):
        super().__init__()
        self.num_features = num_features
        self.height = height
        self.num_classes = num_classes

        num_inner_nodes = (1 << self.height) - 1
        num_leaf_nodes = 1 << self.height

        self.register_parameter(
            "weight",
            nn.Parameter(torch.empty(num_inner_nodes, self.num_features))
        )
        self.register_parameter(
            "bias",
            nn.Parameter(torch.empty(num_inner_nodes))
        )
        self.register_parameter(
            "logits",
            nn.Parameter(torch.empty(num_leaf_nodes, self.num_classes))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        torch.nn.init.kaiming_uniform_(self.logits, a=math.sqrt(5))

    def _path_prob(self, x):
        inner_nodes_prob = torch.sigmoid(
            torch.matmul(x, self.weight.T) + self.bias
        )
        # (batch_size, num_inner_nodes)

        leaf_nodes_prob = []
        for i in range(1 << self.height):
            # enumerate all leaf nodes
            prob = 1
            index = 0
            for j in range(self.height - 1, -1, -1):
                if (i & (1 << j)) == 0:
                    prob = prob * (1 - inner_nodes_prob[:, index])
                    index = index * 2
                else:
                    prob = prob * inner_nodes_prob[:, index]
                    index = index * 2 + 1
            assert i == index
            leaf_nodes_prob.append(prob)

        # (batch_size, num_leaf_nodes)
        return torch.vstack(leaf_nodes_prob).T

    def _dist(self):
        return F.softmax(self.logits, dim=-1)

    def forward(self, x):
        dist = self._dist()
        # (num_leaf_nodes, num_classes)
        path_prob = self._path_prob(x)
        # (batch_size, num_leaf_nodes)

        # (batch_size, num_classes)
        return path_prob.max(dim=-1).values.unsqueeze(-1) * dist.gather(
            dim=0,
            index=path_prob.argmax(dim=-1, keepdim=True)
            .repeat(1, self.num_classes)
        )
