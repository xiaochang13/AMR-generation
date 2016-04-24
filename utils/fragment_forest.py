#!/usr/bin/python
import sys
import time
import pickle
import os
import cPickle
#import alignment
import hypergraph
from fragment_hypergraph import FragmentHGNode, FragmentHGEdge
from amr_graph import *
from amr_utils import *
import logger
import gflags
from HRGSample import *
from rule import Rule
import argparse
from re_utils import *
from collections import defaultdict
from filter_stop_words import *
from lemma_util import initialize_lemma

FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'fragment_nonterminal',
    'X',
    'Nonterminal used for phrase forest.')
gflags.DEFINE_bool(
    'delete_unaligned',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_bool(
    'href',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_integer(
    'max_type',
    7,
    'Set the maximum attachment nodes each nontermial can have.')

FRAGMENT_NT = '[%s]' % FLAGS.fragment_nonterminal
p_rule_f = open('poisoned_rule.gr', 'w')
q_rule_f = open('another_poisoned_rule.gr', 'w')
#unalign_f = open('unaligned_info', 'w')
def filter_with_maxtype(curr_node):
    root_index = curr_node.frag.root
    ext_set = curr_node.frag.ext_set
    nonterm_type = len(ext_set) if root_index in ext_set else (len(ext_set) + 1)
    if nonterm_type > FLAGS.max_type:
        curr_node.set_nosample(True)

#Enlarge a chart with a set of items
def enlarge_chart(prev_chart, new_items):
    for node1 in new_items:
        flag = False
        for node2 in prev_chart:
            if node1.frag == node2.frag:
                for edge in node1.incoming:
                    node2.add_incoming(edge)
                    flag = True

        if not flag: #node1 has never appeared in the previous chart
            prev_chart.add(node1)

#Add one item to a chart
def add_one_item(prev_chart, item):
    flag = False
    for node in prev_chart:
        if node.frag == item.frag:
            for edge in item.incoming:
                node.add_incoming(edge)
                flag = True
    if not flag:
        prev_chart.add(item)

#To verify if a chart item has covered the graph
#i.e. covered all nodes and all edges
def is_goal_item(chart_item):
    fragment = chart_item.frag
    nodes = fragment.nodes
    edges = fragment.edges
    return len(edges) == edges.count()
    #return (len(nodes) == nodes.count()) and (len(edges) == edges.count())

def initialize_edge_alignment(aligned_fragments, edge_alignment):
    for frag in aligned_fragments:
        edge_alignment |= frag.edges

#This method output all the unaligned node information of the current AMR graph
def output_all_unaligned_nodes(edge_alignment, amr_graph):
    un_seq = []
    un_nodes = []
    for i in xrange(len(amr_graph.nodes)):
        curr_node = amr_graph.nodes[i]
        c_edge_index = curr_node.c_edge
        if edge_alignment[c_edge_index] == 0: #Found a concept that is not aligned
            un_seq.append(str(curr_node))
            un_nodes.append(curr_node)
    #print >> unalign_f, ' '.join(un_seq)
    return un_nodes

def output_all_unaligned_edges(edge_alignment, amr_graph):
    for i in xrange(len(edge_alignment)):
        if edge_alignment[i] == 0:
            un_seq.append(str(amr_graph.edges[i]))
    #print >> unalign_f, ' '.join(un_seq)

#Build one node that would cover unaligned edges going out of a node
def build_one_node(curr_frag, curr_start, curr_end, amr_graph, edge_alignment, refine=False):
    curr_node_index = curr_frag.root
    curr_graph_node = amr_graph.nodes[curr_node_index]

    if edge_alignment[curr_graph_node.c_edge] == 0:
        new_node = FragmentHGNode(FRAGMENT_NT, curr_start, curr_end, curr_frag)
        return new_node

    #To remember the unaligned relation going out of each entity
    root_arcs = []
    head_arcs = []
    visited = set()

    is_pred = False #Use for deciding the category of the root node
    is_op = False

    #Dealing with parent edges, ARGx-of
    if len(curr_graph_node.p_edges) > 0:
        for curr_edge_index in curr_graph_node.p_edges:
            curr_edge = amr_graph.edges[curr_edge_index]
            edge_label = curr_edge.label

            if edge_alignment[curr_edge_index] == 1: #This edge has already been aligned
                if curr_frag.edges[curr_edge_index] == 1 and (edge_label[:3] == 'ARG' and 'of' in edge_label):
                    #logger.writeln("what the hell is this")
                    #logger.writeln(str(curr_frag))
                    is_pred = True
                continue

            #Our intuition: ARGs and ops goes with the root
            if (edge_label[:3] == 'ARG' and 'of' in edge_label):
                is_pred = True
                head_arcs.append((curr_edge_index, curr_edge.head))

    if len(curr_graph_node.v_edges) > 0:
        for curr_edge_index in curr_graph_node.v_edges:
            curr_edge = amr_graph.edges[curr_edge_index]
            edge_label = curr_edge.label

            if edge_alignment[curr_edge_index] == 1: #This edge has already been aligned
                if curr_frag.edges[curr_edge_index] == 1 and is_root_arc(edge_label): #Special case, there is already args attached
                    if 'ARG' in edge_label:
                        is_pred = True
                    else:
                        is_op = True
                continue

            tail_node_index = curr_edge.tail

            #Our intuition: ARGs and ops goes with the root
            if is_root_arc(edge_label):
                if 'ARG' in edge_label:
                    is_pred = True
                else:
                    assert 'op' in edge_label
                    is_op = True
                root_arcs.append((curr_edge_index, tail_node_index))

    unaligned_node = None
    if refine:
        init_ext_frag(curr_frag, is_pred, is_op) #Initialize the current fragment

    if len(root_arcs) > 0 or len(head_arcs) > 0:
        n_nodes = len(amr_graph.nodes)
        n_edges = len(amr_graph.edges)
        frag = AMRFragment(n_edges, n_nodes, amr_graph)
        frag.set_root(curr_node_index)

        for rel_index, tail_index in root_arcs:
            edge_alignment[rel_index] = 1
            frag.set_edge(rel_index)
            frag.set_node(tail_index)

        if head_arcs:
            (rel_index, head_index) = head_arcs[0]
            edge_alignment[rel_index] = 1
            frag.set_edge(rel_index)
            frag.set_root(head_index)

        if refine:
            init_ext_frag(frag, is_pred, is_op)

        frag.build_ext_list()
        frag.build_ext_set()
        new_frag = combine_fragments(curr_frag, frag, refine)
        assert new_frag, 'Weird combination found'

        new_node = FragmentHGNode(FRAGMENT_NT, curr_start, curr_end, new_frag)

    else: #Should be either an entity or a single concept
        new_node = FragmentHGNode(FRAGMENT_NT, curr_start, curr_end, curr_frag)

    s = Sample(hypergraph.Hypergraph(new_node), 0)
    new_node.cut = 1
    new_rule, _ = s.extract_one_rule(new_node, None, new_node.frag.ext_list, refine)
    rule_str = '%s ||| %s\n' % (filter_vars(new_rule.dumped_format()), context_str(new_node.frag, amr_graph))
    rule_f.write(rule_str)
    fields = rule_str.split(' ||| ')
    fields[1] = ' '.join(amr_graph.lems[new_node.frag.start: new_node.frag.end])
    lem_rule_str = ' ||| '.join(fields)
    lemma_rule_f.write(lem_rule_str)
    return new_node

def context_str(frag, amr_graph):
    toks = []
    n_toks = len(amr_graph.sent)

    start = frag.start
    end = frag.end
    prev_tok = 'SOS' if start < 1 else amr_graph.sent[start-1]
    prev_2tok = 'SOS' if start < 2 else amr_graph.sent[start-2]
    toks.append(prev_2tok)
    toks.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else amr_graph.sent[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else amr_graph.sent[end+1]
    toks.append(next_tok)
    toks.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else amr_graph.lems[start-1]
    prev_2tok = 'SOS' if start < 2 else amr_graph.lems[start-2]
    toks.append(prev_2tok)
    toks.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else amr_graph.lems[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else amr_graph.lems[end+1]
    toks.append(next_tok)
    toks.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else amr_graph.poss[start-1]
    prev_2tok = 'SOS' if start < 2 else amr_graph.poss[start-2]
    toks.append(prev_2tok)
    toks.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else amr_graph.poss[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else amr_graph.poss[end+1]
    toks.append(next_tok)
    toks.append(next_2tok)
    return ' '.join(toks)

def build_bimap(tok2frags):
    frag2map = defaultdict(set)
    index2frags = defaultdict(set)
    for index in tok2frags:
        for frag in tok2frags[index]:
            index2frags[index].add(frag)
            frag2map[frag].add(index)
            #matched_list = extract_patterns(str(frag), '~e\.[0-9]+(,[0-9]+)*')
            #matched_indexes = parse_indexes(matched_list)
            #for matched_index in matched_indexes:
            #    frag2map[frag].add(matched_index)
    return (index2frags, frag2map)

#Here we try to make the tok to fragment mapping one to one
def rebuild_fragment_map(tok2frags):
    (index2frags, frag2map) = build_bimap(tok2frags)
    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            new_frag_list = []
            min_frag = None
            min_length = 100
            for frag in tok2frags[index]:
                index_set = frag2map[frag]
                assert index in index_set
                if len(index_set) > 1:
                    if len(index_set) < min_length:
                        min_length = len(index_set)
                        min_frag = frag
                    index_set.remove(index)
                else:
                    new_frag_list.append(frag)
            if len(new_frag_list) == 0:
                assert min_frag is not None
                new_frag_list.append(min_frag)
            tok2frags[index] = new_frag_list
    return tok2frags

def extract_fragments(s2g_alignment, amr_graph):
    alignments = s2g_alignment.strip().split()
    tok2frags = defaultdict(list)

    num_nodes = len(amr_graph.nodes)
    num_edges = len(amr_graph.edges)

    op_toks = []
    role_toks = []
    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        span_start = int(curr_tok)
        span_end = span_start + 1

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        frag = AMRFragment(num_edges, num_nodes, amr_graph)
        if index_type == 'c':
            frag.set_root(index)
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' in par_edge.label:
                    op_toks.append((span_start, curr_node.c_edge))

            if curr_node.is_entity():
                role_toks.append((span_start, curr_node.c_edge))

            frag.set_edge(curr_node.c_edge)

        else:
            frag.set_edge(index)
            curr_edge = amr_graph.edges[index]
            frag.set_root(curr_edge.head)
            frag.set_node(curr_edge.tail)

        frag.build_ext_list()
        frag.build_ext_set()

        tok2frags[span_start].append(frag)

    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            tok2frags[index] = connect_adjacent(tok2frags[index], logger)

    tok2frags = rebuild_fragment_map(tok2frags)
    for index in tok2frags:
        for frag in tok2frags[index]:
            frag.set_span(index, index+1)

    return (op_toks, role_toks, tok2frags)

#Verify this fragment contains only one edge and return it
def unique_edge(frag):
    #assert frag.edges.count() == 1, 'Not unify edge fragment found'
    amr_graph = frag.graph
    edge_list = []
    n_edges = len(frag.edges)
    for i in xrange(n_edges):
        if frag.edges[i] == 1:
            edge_list.append(i)
    assert len(edge_list) == frag.edges.count()
    return tuple(edge_list)
    #root_node = amr_graph.nodes[frag.root]
    #for edge_index in root_node.v_edges:
    #    if frag.edges[edge_index] == 1:
    #        return edge_index
    #assert True, 'This is impossible'
    #return None

#def fix_partial_alignments
#def refine_alignment(fragments, amr_graph, der_lemma):
# extract all the binarized fragments combinations for AMR
# each chart item is a set of fragments are consistent with a span of aligned strings
def fragment_decomposition_forest(fragments, amr_graph, unaligned_fragments, edge_alignment, refine=False):
    # save the index mapping so that we can restore indices after phrase
    # decomposition forest generation

    n = len(fragments) #These fragments are aligned, and have some order based on the strings

    global print_sign
    chart = [[set() for j in range(n+1)] for i in range(n+1)]

    start_time = time.time()

    #The leaves of the forest are identified concept fragments
    for i in xrange(n):
        j = i + 1
        frag = fragments[i]

        new_node = build_one_node(frag, i, j, amr_graph, edge_alignment, refine)
        filter_with_maxtype(new_node)
        chart[i][j].add(new_node)

    #These are the unaligned concepts in the graph
    unaligned_nodes = []
    for unaligned_frag in unaligned_fragments:

        unaligned_node = FragmentHGNode(FRAGMENT_NT, -1, -1, unaligned_frag, False, True, True) #Special here
        unaligned_node.cut = 1
        unaligned_nodes.append(unaligned_node)

    edge_to_node = {}
    for i in xrange(n):
        j = i + 1
        curr_candidate = chart[i][j]
        updated = True
        count = 0
        while updated:

            updated = False
            new_node_set = set()
            curr_time = time.time()
            if curr_time - start_time > 30:
                return None
            for node1 in curr_candidate:
                for unaligned_node in unaligned_nodes:
                    #Before combining two fragments, check if they are disjoint
                    if check_disjoint(node1.frag, unaligned_node.frag):
                        (new_frag, connect_frags) = general_combine_fragments(node1.frag, unaligned_node.frag, edge_alignment, refine)
                        if new_frag is None:
                            continue

                        #new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag, False, False, False)
                        new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag, False, False, True)
                        edge = FragmentHGEdge()
                        edge.add_tail(node1)
                        edge.add_tail(unaligned_node)
                        if connect_frags and len(connect_frags) > 0:
                            for unaligned_frag in connect_frags:
                                un_edge_index = unique_edge(unaligned_frag)
                                if un_edge_index not in edge_to_node:
                                    tmp_node = FragmentHGNode(FRAGMENT_NT, -1, -1, unaligned_frag, True, False, False)
                                    edge_to_node[un_edge_index] = tmp_node
                                    tmp_node.cut = 0
                                else:
                                    tmp_node = edge_to_node[un_edge_index]

                                edge.add_tail(tmp_node)

                        new_node.add_incoming(edge)

                        if print_sign:
                            print '%d to %d: %s  %s' % (i, j, ' '.join(new_frag.str_list()), str(new_frag))
                        updated = True
                        filter_with_maxtype(new_node)
                        add_one_item(new_node_set, new_node)
            if updated:
                enlarge_chart(chart[i][j], new_node_set)
                curr_candidate = new_node_set

    start_time = time.time()
    #logger.writeln('Finished dealing with unary')
    for span in xrange(2, n+1):
        for i in xrange(0, n):
            j = i + span
            if j > n:
                continue
            curr_time = time.time()
            if curr_time - start_time > 30:
                return None

            for k in xrange(i+1, j):
                if len(chart[i][k]) == 0 or len(chart[k][j]) == 0:
                    continue
                for node1 in chart[i][k]:
                    for node2 in chart[k][j]:
                        curr_time = time.time()

                        if check_disjoint(node1.frag, node2.frag):
                            #new_frag = combine_fragments(node1.frag, node2.frag)
                            (new_frag, connect_frags) = general_combine_fragments(node1.frag, node2.frag, edge_alignment, refine)

                            if new_frag is None:
                                continue

                            noprint = node1.noprint | node2.noprint
                            new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag, False, False, noprint)

                            children = []
                            children.append(node1)
                            children.append(node2)

                            unaligned_node = None
                            if connect_frags and len(connect_frags) > 0:
                                for unaligned_frag in connect_frags:
                                    un_edge_index = unique_edge(unaligned_frag)
                                    if un_edge_index not in edge_to_node:
                                        tmp_node = FragmentHGNode(FRAGMENT_NT, -1, -1, unaligned_frag, True, False, False)
                                        edge_to_node[un_edge_index] = tmp_node
                                        tmp_node.cut = 0
                                    else:
                                        tmp_node = edge_to_node[un_edge_index]

                                    children.append(tmp_node)


                            if not check_consist(new_node, children):
                                print 'inconsistency here'
                                print str(new_node.frag)
                                print str(node1.frag)
                                print str(node2.frag)

                            edge = FragmentHGEdge()
                            edge.add_tail(node1)
                            edge.add_tail(node2)
                            if connect_frags and len(connect_frags) > 0:
                                for unaligned_frag in connect_frags:
                                    un_edge_index = unique_edge(unaligned_frag)
                                    assert un_edge_index in edge_to_node
                                    #unaligned_node.cut = 0
                                    edge.add_tail(edge_to_node[un_edge_index])

                            new_node.add_incoming(edge)
                            if print_sign:
                                print '%d to %d: %s  %s' % (i, j, ' '.join(new_frag.str_list()), str(new_frag))
                                print '####Children info####'
                                for node in children:
                                    print '%d to %d: %s %s' % (node.frag.start, node.frag.end, ' '.join(node.frag.str_list()) if node.frag.start != -1 else '###', str(node.frag))
                                print '########'

                            s = Sample(hypergraph.Hypergraph(new_node), 0)
                            new_node.cut = 1
                            new_rule, _ = s.extract_one_rule(new_node, None, list(new_node.frag.ext_set))
                            if not new_node.noprint and len(new_node.frag.str_list()) < 8:
                                rule_str = '%s ||| %s\n' % (filter_vars(new_rule.dumped_format()), context_str(new_node.frag, amr_graph))

                                rule_f.write(rule_str)
                                fields = rule_str.split(' ||| ')
                                fields[1] = ' '.join(amr_graph.lems[new_node.frag.start: new_node.frag.end])
                                lem_rule_str = ' ||| '.join(fields)

                                lemma_rule_f.write(lem_rule_str)

                            filter_with_maxtype(new_node)
                            add_one_item(chart[i][j], new_node)

    if print_sign:
        print 'total length is %d' % n
    if chart[0][n] is None or len(chart[0][n]) == 0:
        rule_f.write('\n')
        print '##################################'
        print 'The goal chart is empty, fail to build a goal item'
        print 'Alignment fragments:'
        for frag in fragments:
            print '%s :   %s' % (frag.str_side(), str(frag))
        print 'Unaligned fragments:'
        for frag in unaligned_fragments:
            print str(frag)
        print '#################################'
        return None

    rule_f.write('\n')
    hg = None
    for node in chart[0][n]:
        if is_goal_item(node):
            hg = hypergraph.Hypergraph(node)
            return hg

    #assert hg is not None, 'Failed to build a goal item'
    if hg is None:
        print '##################################'
        print 'No goal item in the final chart'
        print 'Alignment fragments:'
        for frag in fragments:
            print str(frag)

        return None
    return hg

def construct_forest(args):
    amr_graphs = None
    if args.amr_file:
        amr_graphs = load_amr_graphs(args.amr_file)

    if args.parallel:
        global rule_f
        run_nodes = args.nodes.split('+')
        if amr_graphs: #This means the amr graphs are not split yet
            for (i, curr_node) in enumerate(run_nodes):
                curr_graph_file = os.path.join(args.data_dir, 'graph_%d' % i)
                #curr_graph_file = os.path.join(args.data_dir, 'new_graph_%d' % args.sub_id)
                curr_f = open(curr_graph_file, 'wb')
                curr_start = args.sent_per_node * i
                curr_end = args.sent_per_node * (i+1)
                cPickle.dump(amr_graphs[curr_start:curr_end], curr_f)
                curr_f.close()

            if args.dump_graph:
                print 'A total of %d AMR graphs' % len(amr_graphs)
                return

        os.system('rm -rf %s' % args.save_dir)
        os.mkdir(args.save_dir)

        for (i, curr_node) in enumerate(run_nodes):
            cmd = 'python %s --data_dir %s --save_dir %s --sub_id %d --lemma %s --stop %s' % (sys.argv[0], args.data_dir, args.save_dir, i, args.lemma, args.stop)
            print 'start to launch program in %s' % curr_node
            alignment_log_file = os.path.join(args.save_dir, 'alignment_log_%d' % i)

            os.system(r'ssh %s "cd %s; nohup %s >& %s" &' % (curr_node, os.getcwd(), cmd, alignment_log_file))

    else:
        sys.setrecursionlimit(sys.getrecursionlimit() * 30)

        logger.file = open(os.path.join(args.save_dir, 'logger_%d' % args.sub_id), 'w')
        #amr_graph_file = os.path.join(args.data_dir, 'new_graph_%d' % args.sub_id)
        amr_graph_file = os.path.join(args.data_dir, 'graph_%d' % args.sub_id)
        alignment_file = os.path.join(args.data_dir, 'align_%d' % args.sub_id)
        sent_file = os.path.join(args.data_dir, 'sent_%d' % args.sub_id)
        #sent_file = os.path.join(args.data_dir, 'lemma_%d' % args.sub_id)
        lemma_file = os.path.join(args.data_dir, 'lemma_%d' % args.sub_id)
        pos_file = os.path.join(args.data_dir, 'pos_%d' % args.sub_id)
        forest_file = os.path.join(args.save_dir, 'forest_%d' % args.sub_id)
        used_sent_file = os.path.join(args.save_dir, 'used_sent_%d' % args.sub_id)
        dumped_rule_file = os.path.join(args.save_dir, 'dumped_rules_%d' % args.sub_id)
        dumped_lemma_rule_file = os.path.join(args.save_dir, 'lemma_rules_%d' % args.sub_id)

        global rule_f
        rule_f = open(dumped_rule_file, 'w')
        global lemma_rule_f
        lemma_rule_f = open(dumped_lemma_rule_file, 'w')

        amr_f = open(amr_graph_file, 'rb')
        amr_graphs = cPickle.load(amr_f)
        amr_f.close()

        s2g_map = []
        with open(alignment_file, 'r') as align_f:
            for line in align_f:
                s2g_map.append(line.strip())
            align_f.close()

        sents = []
        with open(sent_file, 'r') as sent_f:
            for line in sent_f:
                sents.append(line.strip())
            sent_f.close()

        lemmas = []
        with open(lemma_file, 'r') as lemma_f:
            for line in lemma_f:
                lemmas.append(line.strip())
            lemma_f.close()

        poss = []
        with open(pos_file, 'r') as pos_f:
            for line in pos_f:
                poss.append(line.strip())
            pos_f.close()

        assert len(amr_graphs) == len(s2g_map) and len(amr_graphs) == len(sents) and len(sents) == len(poss), '%d %d %d' % (len(amr_graphs), len(s2g_map), len(sents))

        constructed_forests = []
        sent_indexes = []

        lemma_map = initialize_lemma(args.lemma)

        global print_sign
        stop_words = set([line.strip() for line in open(args.stop, 'r')])

        num_self_cycle = 0
        used_sents = 0

        for (sent_index, (sent, tags, s2g_alignment, amr_graph)) in enumerate(zip(sents, poss, s2g_map, amr_graphs)):

            logger.writeln('Sentence #%d' % sent_index)
            logger.writeln(sent)
            logger.writeln(str(amr_graph))
            logger.writeln(s2g_alignment)

            print_sign = False

            pos_seq = tags.split()
            lemma_seq = lemmas[sent_index].split()

            edge_alignment = bitarray(len(amr_graph.edges))
            if edge_alignment.count() != 0:
                edge_alignment ^= edge_alignment

            assert edge_alignment.count() == 0

            has_cycle = False
            if amr_graph.check_self_cycle():
                num_self_cycle += 1
                has_cycle = True
                logger.writeln('self cycle detected')

            if s2g_alignment == '':
                logger.writeln('totally unaligned')
                continue

            toks = sent.strip().split()
            amr_graph.set_sentence(toks)
            amr_graph.set_lemmas(lemma_seq)
            amr_graph.set_poss(pos_seq)

            aligned_fragments = []

            reentrancies = {}  #Map multiple spans as reentrancies, keeping only one as original, others as connections

            has_multiple = False
            no_alignment = False

            aligned_set = set()

            all_frags = []

            (opt_toks, role_toks, aligned_fragments) = extract_fragments(s2g_alignment, amr_graph)
            logger.writeln(str(opt_toks))
            logger.writeln(str(role_toks))

            if not aligned_fragments:
                logger.writeln('wrong alignments')
                continue

            temp_aligned = set(aligned_fragments.keys())
            aligned_fragments = sorted(aligned_fragments.items(), key=lambda frag: frag[0])

            temp_unaligned = set(xrange(len(pos_seq))) - temp_aligned

            ####Extract entities#####
            #for (frag, root_srcs, opt_srcs) in amr_graph.extract_entities():
            for (frag, frag_label) in amr_graph.extract_entities():
                if len(opt_toks) == 0:
                    logger.writeln("No alignment for the entity found")
                    no_alignment = True

                (frag_start, frag_end, multiple) = extract_entity_spans(frag, opt_toks, role_toks, temp_unaligned)
                if frag_start is None:
                    logger.writeln("No alignment found")
                    logger.writeln(str(frag))

                    no_alignment = True
                    continue

                if multiple:
                    has_multiple = True
                    logger.writeln("Multiple found here!")

                frag.set_span(frag_start, frag_end)

                new_aligned = set(xrange(frag_start, frag_end))
                if len(new_aligned & aligned_set) != 0:
                    has_multiple = True
                    continue

                aligned_set |= new_aligned
                all_frags.append(frag)

                if (edge_alignment & frag.edges).count() != 0:
                    has_multiple = True

                edge_alignment |= frag.edges
                logger.writeln('Exracted entities:')
                logger.writeln(' '.join(toks[frag_start:frag_end]))
                logger.writeln(str(frag))

            if no_alignment:
                continue

            one2many = False
            #####Extra other alignments######
            logger.writeln('Aligned fragments:')
            for (index, frag_list) in aligned_fragments:
                if index in aligned_set:
                    continue

                assert len(frag_list) > 0
                non_conflict = 0
                non_conflict_list = []
                for frag in frag_list:
                    if (edge_alignment & frag.edges).count() == 0:
                        non_conflict += 1
                        non_conflict_list.append(frag)

                if non_conflict != 1:
                    one2many = True

                used_frag = None
                if non_conflict == 0:
                    used_frag = frag_list[0]
                else:
                    used_frag = non_conflict_list[0]

                edge_alignment |= used_frag.edges
                all_frags.append(used_frag)

                aligned_set.add(index)

            logger.writeln("%d aligned edges out of %d total" % (edge_alignment.count(), len(edge_alignment)))
            used_sents += 1

            assert len(toks) == len(pos_seq)
            assert len(toks) == len(lemma_seq)

            unaligned_toks = [(i, tok) for (i, tok) in enumerate(toks) if i not in aligned_set]
            (aligned, unaligned) = amr_graph.recall_unaligned_concepts(edge_alignment, unaligned_toks, lemma_map, stop_words)
            aligned = [x for (x, y, z, k) in aligned]

            all_frags += aligned

            logger.writeln("Retrieved using POS tags and lemmas")
            for frag in aligned:
                logger.writeln(frag.str_side()+ ' :   '+ str(frag))
                for index in xrange(frag.start, frag.end):
                    aligned_set.add(index)

            logger.writeln("Unaligned frags")
            for frag in unaligned:
                logger.writeln(str(frag))

            aligned_fragments = sorted(all_frags, key=lambda frag: (frag.start, frag.end))
            for frag in aligned_fragments:
                frag.build_ext_list()
                frag.build_ext_set()

            unaligned_words = set(range(len(toks))) - aligned_set

            un_seq = []
            for pos in unaligned_words:
                un_seq.append(toks[pos])
                contexts = get_context(toks, lemma_seq, pos_seq, pos, pos+1)
                rule_str = "Nothing ||| %s ||| Nothing ||| Nothing ||| %s\n" % (toks[pos], ' '.join(contexts))
                rule_f.write(rule_str)

            logger.writeln("Unaligned toks: %s" % ' '.join(un_seq))
            #print >> unalign_f, ' '.join(un_seq)

            hg = fragment_decomposition_forest(aligned_fragments, amr_graph, unaligned, edge_alignment, args.refine)

            if has_multiple or no_alignment: #Currently we don't deal with the situation with significant alignment errors
                continue

            if one2many or has_cycle:
                continue

            if hg:
                constructed_forests.append(hg)
                sent_indexes.append(sent_index)

            else:
                logger.writeln("Failed to build a forest")

        forest_f = open(forest_file, 'wb')
        cPickle.dump(constructed_forests, forest_f)
        forest_f.close()
        rule_f.close()
        lemma_rule_f.close()

        used_sent_f = open(used_sent_file, 'wb')
        cPickle.dump(sent_indexes, used_sent_f)
        used_sent_f.close()
        logger.writeln('total used sents is: %d' % used_sents)
        logger.writeln('total dump forests: %d' % len(constructed_forests))
        logger.writeln('finished')

def get_context(toks, lemmas, poss, start, end):
    contexts = []
    n_toks = len(toks)

    prev_tok = 'SOS' if start < 1 else toks[start-1]
    prev_2tok = 'SOS' if start < 2 else toks[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else toks[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else toks[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else lemmas[start-1]
    prev_2tok = 'SOS' if start < 2 else lemmas[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else lemmas[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else lemmas[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else poss[start-1]
    prev_2tok = 'SOS' if start < 2 else poss[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else poss[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else poss[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)
    return contexts

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--dump_graph", action="store_true", help="if only to dump graph object")
    argparser.add_argument("--preprocess", action="store_true", help="if needed to preprocess the AMR graphs into dumped AMR graph objects")
    argparser.add_argument("--parallel", action="store_true", help="if to run multiple process to run the forest construction")
    argparser.add_argument("--dump_rules", action="store_true", help="if to dump lexical rules")
    argparser.add_argument("--refine", action="store_true", help="if to refine the nonterminals")
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--save_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--nodes", type=str, help="nodes for running processes")
    argparser.add_argument("--sent_per_node", type=int, help="number of sentences for each node")
    argparser.add_argument("--sub_id", type=int, help="if a subprocess, the subprocess id")

    args = argparser.parse_args()
    construct_forest(args)
