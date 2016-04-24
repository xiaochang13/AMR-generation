
import operator
import sys,os,collections,random,re
import bleu_score
import numpy
import kenlm
import cPickle
import copy

sys.path.append('/scratch/lsong10/exp.amr_gen/AMR-generation/utils')

from amr_graph import AMRNode,AMREdge,AMRGraph

def dump(f, amr, sent):
    print sent
    for i in range(len(sent)):
        for j in range(i+1, len(sent)):
            # get subamr graph corresponding to sent[i:j]
            # check if it is rooted and connected
            if subamr.is_rooted and subamr.is_connected:
                print >>f, subamr, i, j

if __name__ == '__main__':
    print 'loading reference'
    ref = []
    for line in open('AMR-generation/dev/token','rU'):
        ref.append(line.strip().split())

    i = 0
    f = open('rule_instances.dump','w')
    for line in open('AMR-generation/dev/aligned_amr_nosharp','rU'):
        line = line.strip()
        if len(line) == 0:
            if len(amr_line) > 0:
                amr = AMRGraph(amr_line.strip())
                rst = dump(f, amr, ref[i])
                i += 1
                amr_line = ''
        else:
            assert line.startswith('#') == False
            amr_line = amr_line + line
