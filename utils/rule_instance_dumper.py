
import operator
import sys,os,collections,random,re
import bleu_score
import numpy
import kenlm
import cPickle
import copy

sys.path.append('/scratch/lsong10/exp.amr_gen/AMR-generation/utils')

from amr_graph import AMRNode,AMREdge,AMRGraph
from amr_fragment import *

def dump(f, amr, sent, alignment):
    def recur_dump(cur_root_idx, cur_root, visited):
        max_phr_len = 15
        # init the graph fragment for current subgraph
        n_nodes = len(amr.nodes)
        n_edges = len(amr.edges)
        frag = AMRFragment(n_edges, n_nodes, amr)
        frag.set_root(cur_root_idx)
        frag.set_edge(cur_root.c_edge)
        frag.build_ext_list()
        frag.build_ext_set()
        frag.set_span(1000, -1)
        if ('n', cur_root_idx) in amr_to_sent:
            frag.start = min(amr_to_sent[('n', cur_root_idx)])
            frag.end = max(amr_to_sent[('n', cur_root_idx)])
        if ('e', cur_root.c_edge) in amr_to_sent:
            frag.start = min(amr_to_sent[('n', cur_root_idx)])
            frag.end = max(amr_to_sent[('n', cur_root_idx)])
        # first dump all the children
        new_visited = visited | set([cur_root.get_child(i)[0] for i in range(len(cur_root.v_edges))])
        for i,eid in enumerate(cur_root.v_edges):
            child_idx, child = cur_root.get_child(i)
            if child_idx not in visited:
                child_frag = recur_dump(child_idx, child, new_visited)
                # if one of its children is mapped to a very large span, then stop generating rules for the current span
                if child_frag is None:
                    return None
                merge_child_fragment(frag, child_frag, eid)
                frag.start = min(frag.start, child_frag.start)
                frag.end = max(frag.end, child_frag.end)
        # dump the current subgraph
        if frag.end >= 0:
            # if it is mapped to a very large span, don't generate the rule
            if frag.end - frag.start + 1 > max_phr_len:
                return None
            # check if any word in the aligned span is mapped out of the subgraph
            is_consistent = True
            for i in range(frag.start, frag.end+1):
                if i in sent_to_amr:
                    for (type, id) in sent_to_amr[i]:
                        if (type == 'n' and frag.nodes[id] == 0) or (type == 'e' and frag.edges[id] == 0):
                            is_consistent = False
            print '('+str(frag)+')', '[%d, %d]' % (frag.start, frag.end), is_consistent
            if is_consistent:
                print >>f, '('+str(frag)+')', '|||', '[%d, %d]' % (frag.start, frag.end)
                # incorporating unaligned words on both sides
                for i in range(frag.start-1, -1, -1):
                    if i in sent_to_amr:
                        break
                    for j in range(frag.end+1, len(sent)):
                        if j in sent_to_amr:
                            break
                        if j - i + 1 > max_phr_len:
                            break
                        print >>f, '('+str(frag)+')', '|||', '[%d, %d]' % (i, j)

        return frag

    print >>f, '#SENT:', ' '.join(sent)
    print '#SENT:', ' '.join(sent)
    # process alignment
    amr_to_sent = collections.defaultdict(list)
    sent_to_amr = collections.defaultdict(list)
    for ali in alignment:
        sent_idx, amr_path = ali.split('-')
        sent_idx = int(sent_idx)
        amr_cept = amr.get_from_path(amr_path)
        amr_to_sent[amr_cept].append(sent_idx)
        sent_to_amr[sent_idx].append(amr_cept)
    # dump action
    root = amr.nodes[amr.root]
    recur_dump(amr.root, root, set([amr.root, ]))
    print >>f, ''
    print ''

if __name__ == '__main__':
    print 'loading reference'
    ref = []
    for line in open('AMR-generation/train/token','rU'):
        ref.append(line.strip().split())

    alignment = []
    for line in open('AMR-generation/train/alignment','rU'):
        alignment.append(line.strip().split())

    amr_line = ''
    i = 0
    f = open('train.dump','w')
    for line in open('AMR-generation/train/amr','rU'):
        line = line.strip()
        if len(line) == 0:
            if len(amr_line) > 0:
                amr = AMRGraph(amr_line.strip())
                rst = dump(f, amr, ref[i], alignment[i])
                i += 1
                amr_line = ''
        else:
            assert line.startswith('#') == False
            amr_line = amr_line + line
