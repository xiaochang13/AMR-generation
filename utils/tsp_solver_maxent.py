
from itertools import groupby
import operator
import sys,os,collections,random,re
import bleu_score
import numpy
import cPickle
import copy
import maxent
import bitarray

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

sys.path.append('/home/vax6/p43/mt-scratch2/lsong10/exp.amr_gen/AMR-generation/utils')
sys.path.append('/home/vax6/p43/mt-scratch2/lsong10/exp.amr_gen')

from maxent_feature import subgraph_match, extract_feature
from amr_graph import AMRNode,AMREdge,AMRGraph
import reader

def gen_subgraph_rules(amr, i, rules, rules_match):
    covers = set()
    new_rules = []
    for sub_str, translations in rules.iteritems():
        sub, bit = rules_match[sub_str]
        if bit[i] == True:
            matches = subgraph_match(amr, sub)
            assert len(matches) > 0
            for trans, pos in translations[:int(sys.argv[1])]:
                new_rules.append((sub, trans, pos, matches[0]))
                covers |= matches[0][-1]
    print 'covers: %d total: %d' %(len(covers), len(amr.nodes))
    return new_rules

def gen_concept_rules(amr, concept_trans):
    def get_entity_rule(amr, n, node):
        edge_labels = [amr.edges[x].label for x in node.v_edges]
        nn, nnode = node.get_child(edge_labels.index('name'))
        trans = nnode.get_children_str()
        bit = set([n,nn,]) | nnode.get_all_descendent()
        next = set()
        for i, label in enumerate(edge_labels):
            subn, subnode = node.get_child(i)
            if label == 'wiki':
                bit.add(subn)
                bit = bit | subnode.get_all_descendent()
            else:
                next.add(subn)
        return (bit, trans, next)
    def get_date_rule(amr, n, node):
        labels = [amr.edges[x].label for x in node.v_edges]
        concepts = [node.get_child(i)[1].node_str() for i in range(len(node.v_edges))]
        bit = set([node.get_child(i)[0] for i in range(len(node.v_edges))] + [n,])
        # handle year, month and day
        year = -1
        month = -1
        day = -1
        month_names = ['', 'January','February','March','April','May','June','July','August','September','October','November','December',]
        for i, label in enumerate(labels):
            if label == 'year':
                year = int(concepts[i])
            elif label == 'month':
                month = int(concepts[i])
                concepts[i] = month_names[month]
            elif label == 'day':
                day = int(concepts[i])
                if day in (1, 21, 31,):
                    day_str = str(day) + ' st'
                elif day in (2, 22,):
                    day_str = str(day) + ' nd'
                elif day in (3, 23,):
                    day_str = str(day) + ' rd'
                else:
                    day_str = str(day) + ' th'
        # construct rules
        if year != -1 and month != -1 and day != -1:
            trans = ['%s %s %d' %(month_names[month], day_str, year), ]
        elif year != -1 and month != -1:
            trans = ['%s %d' %(month_names[month], year), ]
        elif month != -1 and day != -1:
            trans = ['%s %s' %(month_names[month], day_str), ]
        else:
            trans = [' '.join(concepts), ' '.join(reversed(concepts)), ]
        return (node.node_str(), trans, None, (n, bit))
    def get_concept_trans(concept):
        pass


    queue = collections.deque()
    queue.append(amr.root)
    visited = set([amr.root,])
    rules = []
    while len(queue) > 0:
        cur = queue.popleft()
        cur_node = amr.nodes[cur]
        if cur_node.is_entity():
            bit, trans, next = get_entity_rule(amr, cur, cur_node)
            assert type(trans) == str
            rules.append((cur_node.node_str(), [trans,], None, (cur, bit)))
            queue.extendleft(next - visited)
            visited |= set(bit)
            visited |= next
        elif cur_node.node_str() == 'date-entity':
            rule = get_date_rule(amr, cur, cur_node)
            rules.append(rule)
            visited |= rule[-1][-1]
        else:
            polarity_vec = [ (amr.edges[e].label == 'polarity' and cur_node.get_child(i)[1].node_str() == '-') for i, e in enumerate(cur_node.v_edges)]
            next = [cur_node.get_child(i)[0] for i, x in enumerate(polarity_vec) if x == False]
            if any(polarity_vec):
                bit = set([cur,polarity_vec.index(True)])
                trans = [x + ' ' + cur_node.node_str_nosuffix() for x in ['does not', 'is not', 'do not', 'am not', 'are not', 'can not',]]
            else:
                bit = set([cur,])
                trans = [cur_node.node_str_nosuffix(),]
            rules.append((cur_node.node_str(), trans, None, (cur, bit)))
            subn_set = set(next)
            subn_set = subn_set - visited
            queue.extendleft(subn_set)
            visited |= subn_set
    return rules

def solve_action(matrix, row_clusters, N, K):
    def distance(from_i, to_j):
        if from_i == to_j:
            return 0
        else:
            if to_j not in matrix[from_i]:
                return K
            return matrix[from_i][to_j]
    # init solver
    routing = pywrapcp.RoutingModel(len(matrix), 1)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    routing.SetDepot(0)
    routing.SetArcCostEvaluatorOfAllVehicles(distance)
    # add disjunction
    for nid, disj in row_clusters.iteritems():
        routing.AddDisjunction(disj)
    # remove unnecessary connections:
    #   0 -> N-1
    #   N-1 -> [1,...,N-2]
    #   [1,...,N-2] -> 0
    routing.NextVar(0).RemoveValue(N-1)
    for i in range(1, N-1):
        routing.NextVar(N-1).RemoveValue(i)
        routing.NextVar(i).RemoveValue(0)
    # solve
    assignment = routing.Solve()
    if assignment:
        node = routing.Start(0)
        route = [node,]
        while node < N and not routing.IsEnd(node):
            node = assignment.Value(routing.NextVar(node))
            route.append(node)
        print '+++++++++++++++++++++'
        print assignment.ObjectiveValue()
        print route
        print '---------------------'
        return (assignment.ObjectiveValue(), route,)
    else:
        print 'No answer !!!'
        return None


def solve_by_tsp(amr, iii, rules, rules_match, concept_trans, model):
    def model_score(rule1, rule2):
        feats = extract_feature(amr, rule1, rule2)
        cost = model.eval(feats.keys(), 'n') * 100
        #print rule1
        #print rule2
        #print feats.keys()
        #print '=========== cost', cost
        return cost
    def is_rule_head(i):
        return row_string[i] != ''
    def is_rule_tail(i):
        return row_string[i+1] != ''

    K = 10000
    matched = []
    matched.extend(gen_concept_rules(amr, concept_trans))
    print 'concept rule number', len(matched)
    matched.extend(gen_subgraph_rules(amr, iii, rules, rules_match))
    print ' + subgraph rule number', len(matched)

    # each row index is represented as (ni, pi, string)
    # pi is a set of ni involved in the phrase
    row_string = ['<s>',]
    row_ruleid = [-1,]
    disjunctions = collections.defaultdict(list)
    # from subgraph rules
    for i, (sub, trans, pos, (n, bit)) in enumerate(matched):
        mark = len(row_string)
        row_string.append(' '.join(trans))
        row_ruleid.append(i)
        disjunctions[n].append(len(row_string)-1)
        for subn in bit:
            if subn != n:
                row_string.append('')
                row_ruleid.append(i)
                disjunctions[subn].append(len(row_string)-1)
    row_string.append('</s>')
    row_ruleid.append(-2)
    N = len(row_string)

    start_rule = ('<s>', ['<s>',], ['<s>',], None)
    end_rule = ('</s>', ['</s>',], ['</s>',], None)

    # init matrix
    matrix = {}
    for i in range(N):
        matrix[i] = {}
    print 'matrix size N*N where N is ', N
    # build tsp
    # START -> clusters
    for i in range(1, N-1):
        if is_rule_head(i):
            matrix[0][i] = model_score(start_rule, matched[row_ruleid[i]])
        else:
            matrix[0][i] = K
    # Cluster -> END
    for i in range(1, N-1):
        if is_rule_tail(i):
            matrix[i][N-1] = model_score(matched[row_ruleid[i]], end_rule)
        else:
            matrix[i][N-1] = K
    # END -> START
    matrix[N-1][0] = 0
    # Cluster -> Cluster
    for i in range(1,N-1):
        for j in range(1,N-1):
            if i == j:
                matrix[i][j] = 0
            elif row_ruleid[i] == row_ruleid[j]:
                if i+1 == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = K
            else:
                intersect = matched[row_ruleid[i]][-1][-1] & matched[row_ruleid[j]][-1][-1]
                if is_rule_tail(i) and is_rule_head(j) and len(intersect) == 0:
                    matrix[i][j] = model_score(matched[row_ruleid[i]],matched[row_ruleid[j]])
                else:
                    matrix[i][j] = K

    # solve action
    ret = solve_action(matrix, disjunctions, N, K)
    # build generated string
    # if no answer, then output all concept in breadth first search
    if ret == None:
        return ['<s>',] + [x for x in groups.iterkeys()] + ['</s>',]
    else:
        result = []
        (cost,route) = ret
        for i in route:
            if 0 < i and i < N-1 and len(row_string[i]) > 0:
                result += row_string[i].split()
        return result


if __name__ == '__main__':
    print 'loading rules'
    rules, rules_match = cPickle.load(open('train.amr.rm_red.filter_dev_v2.cp','rb')) # each rule is (sub, trans, pos, bitarray)
    for sub_str, translist in rules.iteritems():
        freq = collections.Counter([' '.join(trans) for trans, pos in translist])
        translist.sort(key=lambda x: -1.0*freq[' '.join(x[0])])

    print 'loading maxent model'
    LM = maxent.MaxentModel()
    LM.load('features.txt.balance.model')

    print 'loading concept trans'
    concept_trans = collections.defaultdict(set)
    for line in open('verbalization-list-v1.06.txt','rU'):
        items = line.strip().split()
        if items[0] != 'DO-NOT-VERBALIZE':
            trans = items[1]
            i = items.index('TO')
            root = items[i+1]
            if len(items) > i+2:
                concept_trans[' '.join(items[i+1:i+4])].add(trans)
            else:
                concept_trans[items[i+1]].add(trans)

    print 'loading reference'
    sent = []
    for i, sss in reader.read_sent('AMR-generation/dev/token.lower'):
        sent.append(sss)

    print 'solving'
    f = open('log.bleu','w')
    f = sys.stdout
    ans = []
    ref = []
    ccc = 0
    for i, amr in reader.read_amr('AMR-generation/dev/aligned_amr_nosharp'):
        if i in (7,8,) or len(sent[i]) > 50:
            continue
        ref.append([sent[i],])
        rst = solve_by_tsp(amr, i, rules, rules_match, concept_trans, LM)
        rst = [x.lower() for x in rst]
        ans.append(rst)
        print >>f, 'SENT: (%d,%d)' %(i,ccc), ' '.join(rst)
        print >>f, 'BLEU for Sentence %d:' % i, bleu_score.sentence_bleu(ref[-1], ans[-1])
        print >>f, 'Corpus BLEU', bleu_score.corpus_bleu(ref, ans)
        ccc += 1
    print 'Finished decoding'

