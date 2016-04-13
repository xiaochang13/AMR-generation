
import operator
import sys,os,collections,random,re
import bleu_score
import numpy
import kenlm
import cPickle
import copy

sys.path.append('/scratch/lsong10/exp.amr_gen/AMR-generation/utils')

from amr_graph import AMRNode,AMREdge,AMRGraph

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# prune rules for debuging
def rule_pruning(groups):
    new_groups = {}
    for (cept, cand) in groups.iteritems():
        if len(cand) > 4:
            new_groups[cept] = cand[:4]
        else:
            new_groups[cept] = cand
        if len(new_groups) > 2:
            break
    return new_groups


def recursive_subgraph_match(amr_node, sub_node, amr_visited, sub_visited):
    if sub_node.is_leaf():
        return True
    c1 = amr_node.get_unvisited_children(amr_visited)
    c2 = sub_node.get_unvisited_children(sub_visited)
    if len(c1) != len(c2):
        return False
    if all([ c1[i][1] == c2[i][1] for i in range(len(c1)) ]):
        amr_visited.update([x[0] for x in c1])
        sub_visited.update([x[0] for x in c2])
        return all([ recursive_subgraph_match(c1[i][2], c2[i][2], amr_visited, sub_visited) for i in range(len(c1)) ])
    else:
        return False


def gen_subgraph_rules(amr, phrases):
    groups = collections.defaultdict(list)
    # each rule
    for phr in phrases:
        [itemg, items, ] = [x.strip() for x in phr.split('|||')]
        sub = AMRGraph(itemg)
        # each subgraph
        for i,node in enumerate(amr.nodes):
            if node.node_str_nosuffix() == sub.nodes[sub.root].node_str_nosuffix():
                amr_visited = set([i,])
                sub_visited = set([sub.root,])
                if recursive_subgraph_match(node, sub.nodes[sub.root], amr_visited, sub_visited):
                    # we find a match,
                    for nid in amr_visited:
                        groups[nid].append(items)
    print 'subgraph rules:', len(groups)
    return groups


def gen_action(amr, id, naive, naive_dict, groups):
    def gen_entity(node):
        i = [node.graph.edges[x].label for x in node.v_edges].index('name')
        cid, cn = node.get_child(i)
        return cn.get_children_str()
    def gen_date(node):
        pass

    node = amr.nodes[id]
    edge = amr.edges[node.c_edge]
    if node.is_entity():
        rst = gen_entity(node)
        groups[id] = [rst,]
    elif edge.label in amr.dict:
        concept = amr.dict[edge.label].split('-')[0]
        cand = [concept,]
        # consider chunk rule now
        #if concept in naive_dict:
        #    cand += [naive[x] for x in naive_dict[concept]]
        groups[id] = cand
    elif edge.label not in ('',):
        cand = [edge.label,]
        groups[id] = cand


def gen_naive_rules(amr, naive, naive_dict):
    queue = collections.deque()
    queue.append(amr.root)
    groups = {}
    gen_action(amr, amr.root, naive, naive_dict, groups)
    while len(queue) > 0:
        curr = queue.popleft()
        curr_node = amr.nodes[curr]
        # for some type of nodes, we process the entire subgraph rooted by it
        if curr_node.is_entity():
            continue
        children = curr_node.get_unvisited_children(groups, is_sort = False)
        for (cid, cstr, cnode) in children:
            queue.append(cid)
            gen_action(amr, cid, naive, naive_dict, groups)
    print 'naive rules:', len(groups)
    return groups

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
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    routing.SetDepot(0)
    routing.SetArcCostEvaluatorOfAllVehicles(distance)
    # add disjunction
    for st,ed in row_clusters:
        routing.AddDisjunction(range(st,ed))
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


def solve_by_tsp(amr, phrases, naive, naive_dict, LM):
    def group_update(g1, g2):
        for k,v in g2.iteritems():
            if k in g1:
                g1[k].update(v)
            else:
                g1[k] = set(v)
    def prev_in_clus(st, ed, i):
        i_prime = ed-1 if i == st else i-1
        return i_prime

    K = 10000
    groups = {}
    group_update(groups, gen_naive_rules(amr, naive, naive_dict))
    #group_update(groups, gen_subgraph_rules(amr, phrases))
    # init matrix
    matrix = {0:{},}
    row_string = ['<s>',]
    row_clusters = []     # only keep the span of the cluster
    for (concept, candidates) in groups.iteritems():
        st = len(row_string)
        for cand in candidates:
            row_string.append(cand)
            matrix[len(row_string)-1] = {}
        row_clusters.append((st, len(row_string), ))
    row_string.append('</s>')
    matrix[len(row_string)-1] = {}
    N = len(row_string)
    print 'matrix size N*N where N is ', N
    print row_clusters
    # build tsp
    # START -> clusters
    for i in range(1, N-1):
        aaa = row_string[i].split()[0]
        matrix[0][i] = -100.0*LM.score(aaa)
    # END -> START
    matrix[N-1][0] = 0
    # for out-going edge, switch the from node to the previous node
    # Cluster -> END
    for (st,ed) in row_clusters:
        for i in range(st, ed):
            i_prime = prev_in_clus(st, ed, i)
            bbb = row_string[i].split()[-1]
            matrix[i_prime][N-1] = -100.0*LM.score(bbb)
    # Cluster -> Cluster
    for (st1, ed1) in row_clusters:
        # inner
        for i in range(st1, ed1):
            for j in range(st1, ed1):
                if i == j:
                    matrix[i][j] = 0
                elif prev_in_clus(st1, ed1, j) == i:
                    matrix[i][j] = 0
        # outer
        for (st2, ed2) in row_clusters:
            # skip identical clusters
            if st1 == st2 and ed1 == ed2:
                continue
            for i in range(st1, ed1):
                for j in range(st2, ed2):
                    aaa = row_string[i].split()[-1]
                    bbb = row_string[j].split()[0]
                    i_prime = prev_in_clus(st1, ed1, i)
                    matrix[i_prime][j] = 100.0*(LM.score(aaa) - LM.score(aaa + ' ' + bbb)) # -log p(bbb|aaa)
    # solve action
    ret = solve_action(matrix, row_clusters, N, K)
    # build generated string
    # if no answer, then output all concept in breadth first search
    if ret == None:
        return ['<s>',] + [x for x in groups.iterkeys()] + ['</s>',]
    else:
        # make row of cluster id, for fast compute
        row_cluster_id = [0,]
        cur_id = 1
        for (st, ed) in row_clusters:
            row_cluster_id += [cur_id,]*(ed-st)
            cur_id += 1
        row_cluster_id.append(cur_id)
        # generate result
        result = []
        (cost,route) = ret
        for i in route:
            if 0 < i and i < N-1:
                result += row_string[i].split()
        return result
        # old version
        #visited_id = set()
        #last_id = -1
        #for i in route[1:-1]:
        #    i = int(i)
        #    id = row_cluster_id[i]  # id of group
        #    if id != last_id:
        #        result.append(row_string[i]) # get trans (i), not group (id)
        #        if id not in visited_id:
        #            visited_id.add(last_id)
        #        else:
        #            print 'get away from a group', i
        #    last_id = id
        #assert len(groups) <= len(result)
        return result

if __name__ == '__main__':
    print 'loading phrase table'
    phrases = set()
    discarded = 0
    for line in open('train.amr','rU'):
        if len(line.strip()) > 0 and line[0][0] == '(':
            line = re.sub('-[0-9]+ ', ' ', line.strip())
            try:
                tmp = AMRGraph(line.split('|||')[0].strip())
                phrases.add(line)
            except Exception:
                discarded += 1
    print 'len(phrases)', len(phrases)
    print 'discarded', discarded

    print 'loading relations'
    rel_count, rel_trans_count = cPickle.load(open('train.rel.cp','rb'))
    rel2trans = {}
    for rel, subdict in rel_trans_count.iteritems():
        trans = sorted(subdict.iteritems(), key=lambda item: -item[1])
        trans = [x for x, y in trans[:10]]
        rel2trans[rel] = trans
        print rel, '-->', trans

    print 'loading bi-gram chunk rules'
    naive = []
    naive_dict = collections.defaultdict(list)
    for i,line in enumerate(open('train.chk.2','rU')):
        phr = line.strip()
        naive.append(phr)
        for token in phr.split():
            naive_dict[token].append(i)
    print 'len(naive)', len(naive)
    print 'len(naive_dict)', len(naive_dict)

    print 'loading LM'
    LM = kenlm.LanguageModel('train.token.lm.arpa')
    print '%d-gram model' % LM.order

    print 'solving'
    ans = []
    amr_line = ''
    for line in open('AMR-generation/dev/aligned_amr_nosharp','rU'):
    #for line in open('debug.amr','rU'):
        line = line.strip()
        if len(line) == 0:
            if len(amr_line) > 0:
                amr = AMRGraph(amr_line.strip())
                rst = solve_by_tsp(amr, phrases, naive, naive_dict, LM)
                ans.append(rst)
                print 'SENT: ', ' '.join(rst)
            amr_line = ''
        else:
            assert line.startswith('#') == False
            amr_line = amr_line + line

    ref = []
    for line in open('AMR-generation/dev/token','rU'):
    #for line in open('debug.tok','rU'):
        ref.append([line.strip().split(),])

    print 'Finished decoding len(ans)', len(ans), 'len(ref)', len(ref)

    cPickle.dump((ans,ref), open('result.cp','wb'))
    # calc BLEU score
    for i in range(len(ans)):
        print 'BLEU for Sentence %d:' % i, bleu_score.sentence_bleu(ref[i], ans[i])
    print 'Corpus BLEU', bleu_score.corpus_bleu(ref, ans)
