#!/usr/bin/env python3

import sys
import time
import os

from hypergraph import Node, Edge
#from consensus_training import cartesian
#import rule_filter

class FragmentHGNode(Node):
    def __init__(self, nt, fi, fj, fragment, _nosample=False, _poison=False, _noprint=False):
        Node.__init__(self)
        self.nt = nt
        self.frag = fragment
        self.start = fi
        self.end = fj
        self.edge = -1
        self.cut = 1
        self.nosample = _nosample
        self.poisoned = _poison
        self.noprint = _noprint

    def __str__(self):
        return str(self.frag)

    def set_nosample(self, _nosample=True):
        self.nosample = _nosample

    def __eq__(self, other):
        if (self.start != other.start) or (self.end != other.end):
            return False
        return (self.frag == other.frag)

    def __hash__(self):
        return hash(str(self.frag))

class FragmentHGEdge(Edge):
    def __init__(self, rule=None):
        Edge.__init__(self)
        self.rule = rule

    def __str__(self):
        return '%s' % self.rule

    def serialize(self):
        edge_str = Edge.serialize(self)
        rule_str = str(self.rule)
        return ' ||||| '.join([edge_str, rule_str])

    def deserialize(self):
        edge_str, rule_str = line.split('|||||')
        tail_ids, head_id = Edge.deserialize(self, edge_str)
        self.rule = Rule()
        self.rule.fromstr(rule_str)
        return tail_ids, head_id
