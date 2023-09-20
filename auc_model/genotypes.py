from collections import namedtuple

Genotype = namedtuple('Genotype', 's, s2, x1, x2, select, fuse')

PRIMITIVES1 = [
    'identity',
    'ffn',
    'interaction_1',
    'interaction_2',
    'interaction_3'
]

PRIMITIVES2 = [
    'identity',
    'conv',
    'attention',
    'rnn',
    'ffn',
    'interaction_2',
]

PRIMITIVES3 = [
    'identity',
    'zero'
]

PRIMITIVES4 = [
    'sum',
    'att',
    'mlp',
]