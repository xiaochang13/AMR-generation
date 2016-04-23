
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
    groups = {}
    matched = []
    # each rule
    for itemg, sublist in phrases.iteritems():
        for items in sublist:
            sub = AMRGraph(itemg)
            # each subgraph
            for i,node in enumerate(amr.nodes):
                if node.node_str_nosuffix() == sub.nodes[sub.root].node_str_nosuffix():
                    amr_visited = set([i,])
                    sub_visited = set([sub.root,])
                    if recursive_subgraph_match(node, sub.nodes[sub.root], amr_visited, sub_visited):
                        # we find a subgraph match,
                        if len(amr_visited) > 1:
                            print 'match large subgraph!!!', amr_visited, items
                            matched.append((i, amr_visited, items))
                        ## there is only one node, should be i
                        #else:
                        #    if i not in groups:
                        #        groups[i] = [items, ]
                        #    else:
                        #        groups[i].append(items)
    #print 'number of subgraph rule matches:', len(matched)
    return groups, matched


def gen_naive_action(amr, id, naive, naive_dict, groups):
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
        ## consider chunk rule now
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
    gen_naive_action(amr, amr.root, naive, naive_dict, groups)
    while len(queue) > 0:
        curr = queue.popleft()
        curr_node = amr.nodes[curr]
        # for some type of nodes, we process the entire subgraph rooted by it
        if curr_node.is_entity():
            continue
        children = curr_node.get_unvisited_children(groups, is_sort = False)
        for (cid, cstr, cnode) in children:
            queue.append(cid)
            gen_naive_action(amr, cid, naive, naive_dict, groups)
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


def solve_by_tsp(amr, phrases, naive, naive_dict, LM):
    def group_update(g1, g2):
        for k,v in g2.iteritems():
            if k in g1:
                g1[k].update(v)
            else:
                g1[k] = set(v)
    #def prev_in_clus(st, ed, i):
    #    print 'assume all nodes in a cluster are together, be careful for subgraph rules'
    #    i_prime = ed-1 if i == st else i-1
    #    return i_prime
    def subg_head(i, row_string, M):
        if i < M:
            return i
        while row_string[i] == '':
            i -= 1
        assert i >= M
        return i
    def subg_tail(i, row_string, M, N):
        if i < M:
            return i
        while i < N-1 and row_string[i+1] == '':
            i += 1
        assert row_string[i] == '' and row_string[i+1] != ''
        return i
    def LM_score(i, j, M, row_string, LM):
        i = subg_head(i, row_string, M)
        j = subg_head(j, row_string, M)
        aaa = row_string[i].split()[-1]
        bbb = row_string[j].split()[0]
        return 100.0*(LM.score(aaa) - LM.score(aaa + ' ' + bbb)) # -log p(bbb|aaa)

    K = 10000
    groups = {}
    ggg = gen_naive_rules(amr, naive, naive_dict)
    group_update(groups, ggg)
    ggg, matched = gen_subgraph_rules(amr, phrases)
    group_update(groups, ggg)

    # each row index is represented as (ni, pi, string)
    # pi is a set of ni involved in the phrase
    row_string = ['<s>',]
    row_ni = [-1,]
    row_pi = [set(),]
    disjunctions = {}
    # from naive rules
    for (nid, candidates) in groups.iteritems():
        st = len(row_string)
        for cand in candidates:
            row_string.append(cand)
            row_ni.append(nid)
            row_pi.append(set([nid,]))
        disjunctions[nid] = range(st, len(row_string))
    M = len(row_string)
    #tmpM = M
    # from subgraph rules
    for (nid, idset, trans) in matched:
        #print 'subg:',tmpM, tmpM+len(idset)-1
        #tmpM = tmpM+len(idset)
        row_string.append(trans)
        row_ni.append(nid)
        row_pi.append(idset)
        if nid not in disjunctions:
            disjunctions[nid] = []
        disjunctions[nid].append(len(row_string)-1)
        for id in idset:
            if id != nid:
                row_string.append('')
                row_ni.append(id)
                row_pi.append(idset)
                if id not in disjunctions:
                    disjunctions[id] = []
                disjunctions[id].append(len(row_string)-1)
    row_string.append('</s>')
    row_ni.append(-1)
    row_pi.append(set())
    N = len(row_string)

    row_head = [0,]
    row_tail = [0,]
    for i in range(1, N-1):
        row_head.append(subg_head(i,row_string,M))
        row_tail.append(subg_tail(i,row_string,M,N))
    row_head.append(N-1)
    row_tail.append(N-1)

    # init matrix
    matrix = {}
    for i in range(N):
        matrix[i] = {}
    print 'matrix size N*N where N is ', N, 'M is ', M
    # build tsp
    # START -> clusters
    #    single-node rules, head of subgraph rules: LM
    #    middle & tail nodes of subgraph rules: inf
    for i in range(1, N-1):
        if i >= M and row_string[i] == '':
            matrix[0][i] = K
        else:
            aaa = '<s> ' + row_string[i].split()[0]
            matrix[0][i] = -100.0*LM.score(aaa)
    # END -> START
    matrix[N-1][0] = 0
    # Cluster -> END
    #   single-node rules, tail of subgraph rules --> LM
    #   head & middle nodes of subgraph rules: inf
    for i in range(1,N-1):
        if i < M or (row_string[i] == '' and row_string[i+1] != ''):
            bbb = row_string[row_head[i]]
            bbb = bbb.split()[-1] + ' </s>'
            matrix[i][N-1] = -100.0*LM.score(bbb)
        else:
            matrix[i][N-1] = K
    # Cluster -> Cluster
    for i in range(1,N-1):
        for j in range(1,N-1):
            if i == j:
                matrix[i][j] = 0
                continue
            # they should be in the same disjunction, whatever value is Okay
            if row_ni[i] == row_ni[j]:
                assert i in disjunctions[row_ni[i]] and j in disjunctions[row_ni[i]]
                matrix[i][j] = 0
            elif row_head[i] == row_head[j]:
                if i+1 == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = K
            else:
                sect = row_pi[i] & row_pi[j]
                if row_tail[i] == i and row_head[j] == j and len(sect) == 0:
                    matrix[i][j] = LM_score(i, j, M, row_string, LM)
                else:
                    matrix[i][j] = K

    #mat = numpy.full((N,N),K)
    #for i,sub in matrix.iteritems():
    #    for j,val in sub.iteritems():
    #        mat[i][j] = val
    #numpy.set_printoptions(threshold='nan')
    #print mat
    #import matplotlib.pylab as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.set_aspect('equal')
    #plt.imshow(mat, interpolation="nearest")
    #plt.colorbar()
    #plt.show()

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
    print 'loading phrase table'
    amr_trans_count = cPickle.load(open('train.amr.filtered.cp','rb'))
    # keep top 10
    phrases = {}
    for amr, subdict in amr_trans_count.iteritems():
        trans = sorted(subdict.iteritems(), key=lambda item: -item[1])
        trans = [x for x, y in trans[:10]]
        phrases[amr] = trans


    print 'loading relations'
    rel_count, rel_trans_count = cPickle.load(open('train.rel.cp','rb'))
    rel2trans = {}
    for rel, subdict in rel_trans_count.iteritems():
        trans = sorted(subdict.iteritems(), key=lambda item: -item[1])
        trans = [x for x, y in trans[:10]]
        rel2trans[rel] = trans


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

    print 'loading reference'
    ref = []
    for line in open('AMR-generation/dev/token','rU'):
        ref.append([line.strip().split(),])

    f = open('log.bleu','w')
    print 'solving'
    ans = []
    amr_line = ''
    i = 0
    for line in open('AMR-generation/dev/aligned_amr_nosharp','rU'):
        line = line.strip()
        if len(line) == 0:
            if len(amr_line) > 0:
                amr = AMRGraph(amr_line.strip())
                rst = solve_by_tsp(amr, phrases, naive, naive_dict, LM)
                ans.append(rst)
                print >>f, 'SENT: ', ' '.join(rst)
                print >>f, 'BLEU for Sentence %d:' % i, bleu_score.sentence_bleu(ref[i], rst)
                print >>f, 'Corpus BLEU', bleu_score.corpus_bleu(ref[:i+1], ans)
                i += 1
                amr_line = ''
        else:
            assert line.startswith('#') == False
            amr_line = amr_line + line
    print 'Finished decoding'

    print 'dumping result'
    cPickle.dump((ans,ref), open('result.cp','wb'))

