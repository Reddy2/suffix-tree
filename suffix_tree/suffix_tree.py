from collections import Counter


class SuffixTreeNode:
    leaf_end = -1

    # TODO: Only used for visualization during construction right now.  We can just use dfs_num later
    num_nodes_created = 0
    
    def __init__(self, start, end=None, string_id=None, start_idx=None):
        self.children = {}
        self.start = start
        self._end = end
        self.suffix_link = None
        self.string_id = string_id
        self.string_ids = [string_id]
        self.start_idxs = [start_idx]
        self.parent = None

        # TODO: See note about num_nodes_created
        self.num = str(SuffixTreeNode.num_nodes_created)
        SuffixTreeNode.num_nodes_created += 1
        

    @property
    def end(self):
        return self._end if self._end is not None else SuffixTreeNode.leaf_end


    @end.setter
    def end(self, val):
        self._end = val
        

    def edge_length(self):
        return self.end + 1 - self.start
        


class SuffixTree:
    def __init__(self, strings):
        # TODO: Allow single strings
        self._string = strings[0]
        self._string_id = 0
        self._strings = {i: strings[i] + '$' for i in range(len(strings))}  # TODO: Probably add something to ER3 rather than do + '$'
        self._start_idx = 0
        self._root = SuffixTreeNode(-1, -1)
        self._active_node = self._root
        self._active_edge = ''
        self._active_edge_index = 0
        self._active_length = 0
        self._phase = 0
        self._remainder = 0
        self._need_suffix_link = None
        self._shared_phase = True
        self._string_leaves = []
        self._terminal_character = '$'
        self._terminal_er3 = False

        for string in strings:
            self.add_string(string)

        # This will also number the nodes and assign their parents
        self._preprocess_lca()


    def _active_edge(self):
        return text[self.active_edge]


    def _add_suffix_link(self, node):
        if self._need_suffix_link:
            self._need_suffix_link.suffix_link = node
        self._need_suffix_link = node


    def _walk_down(self, next_node):
        edge_length = next_node.edge_length()
        if self._active_length >= edge_length:
            self._active_length -= edge_length
            self._active_edge_index += edge_length
            self._active_edge = self._string[self._active_edge_index]
            self._active_node = next_node
            return True

        return False


    def _add_leaf(self):
        leaf = SuffixTreeNode(self._phase, None, self._string_id, self._start_idx)
        self._string_leaves.append(leaf)
        self._start_idx += 1
        return leaf


    def _add_char(self, c):
        # This is the main part of Ukkonen's Algorithm (the Single Phase Algorithm mixed in with the Single Extension Algorithm from Gusfield's book)
        # See the bottom of the file for an in-depth commented version of this function (that is probably not fully correct yet)
        
        SuffixTreeNode.leaf_end = self._phase

        self._need_suffix_link = None
        self._remainder += 1
        while self._remainder > 0:
            if self._active_length == 0:
                self._active_edge_index = self._phase
                self._active_edge = self._string[self._active_edge_index]
            
            next_node = self._active_node.children.get(self._active_edge)
            if next_node:
                if self._walk_down(next_node):
                    continue
                elif self._strings[next_node.string_id][next_node.start + self._active_length] == c:
                    if c == self._terminal_character:
                        next_node.string_ids.append(self._string_id)
                        next_node.start_idxs.append(self._start_idx)
                        self._start_idx += 1
                        if not self._terminal_er3:
                            self._add_suffix_link(self._active_node)
                            self._terminal_er3 = True
                    else:
                        self._active_length += 1
                        self._add_suffix_link(self._active_node)
                        break
                else:
                    split_node = SuffixTreeNode(next_node.start, next_node.start + self._active_length - 1, next_node.string_id)
                    self._active_node.children[self._active_edge] = split_node
                    leaf = self._add_leaf()
                    split_node.children[c] = leaf
                    next_node.start += self._active_length
                    split_node.children[self._strings[next_node.string_id][next_node.start]] = next_node
                    self._add_suffix_link(split_node)
            else:
                leaf = self._add_leaf()
                self._active_node.children[self._active_edge] = leaf
                self._add_suffix_link(self._active_node)

            if self._active_node == self._root and self._active_length > 0:
                self._active_edge_index += 1
                self._active_edge = self._string[self._active_edge_index]
                self._active_length -= 1
            elif self._active_node != self._root:
                self._active_node = self._active_node.suffix_link
                
            self._remainder -= 1
            
        self._phase += 1
                

    def add_string(self, string):
        # Note that the active point/remainder properly resets after each string is inserted due to the terminal character being added last
        self._string = string + self._terminal_character
        self._phase = 0
        self._terminal_er3 = False
        self._start_idx = 0
        for c in self._string:
            self._add_char(c)

        # Any newly added leaves for the current string are set to their final values (we no longer use the value 'e' described in the book)
        for leaf in self._string_leaves:
            leaf.end = len(self._string) - 1
        self._string_leaves.clear()

        self._string_id += 1


    def _preprocess_lca(self):
        self._lca = LCA(self._root)


    def lca(self, x, y):
        return self._lca.lca(x, y)


    def _find_node(self, s):
        node = self._root
        c = s[0]
        i = 0
        while True:
            # Checks 0 index
            node = node.children.get(c)
            if not node:
                return []
            if i == len(s) - 1:
                return node
            i += 1
            c = s[i]

            # Check from index 1 to node.end
            # for j in range(node.start + 1, node.end + 1)
            j = 1
            while i < len(s) and j < node.edge_length():
                if c != self._strings[node.string_id][node.start + j]:
                    return []
                if i == len(s) - 1:
                    return node
                
                i += 1
                j += 1
                c = s[i]


    def _leaves_of_node(self, node):
        leaves = []
        def f(node):
            if not node.children:
                leaves.append(node)

            for child in node.children.values():
                f(child)

        f(node)
        return leaves
                

    def find(self, s):
        node = self._find_node(s)
        if not node:
            return []
        
        leaves = self._leaves_of_node(node)
        ids_and_indexes = []
        for leaf in leaves:
            ids_and_indexes += list(zip(leaf.string_ids, leaf.start_idxs))
        return ids_and_indexes


    def lcs(self):
        # Gusfield page 205 (9.7)
        # Original paper: Color Set Size Problem with Applications to String Matching 
        #  Link: http://sci-hub.tw/https://doi.org/10.1007/3-540-56024-6_19
        # Note: There may be an updated version and even an updated paper by the same author ?
        # Note: Our suffix tree uses the same terminal string identifier for all strings,
        #        so this requires some minor modifications to the given algorithm
        
        # Step 2/Step 3
        # Note that no explicit leaf number is needed to be stored to add the leaf to L in the proper order
        L = [[] for i in range(len(self._strings))]
        def dfs_num(node):
            if not node.children:
                for string_id in node.string_ids:
                    L[string_id].append(node)

            for child in node.children.values():
                dfs_num(child)

        dfs_num(self._root)

        # Step 4/5
        h = Counter()   # Note nodes not in h will have a count of 0
        for i in range(len(L)):
            for j in range(len(L[i]) - 1):
                lca = self.lca(L[i][j], L[i][j + 1])
                h[lca] += 1

        # Step 6/7
        C = {}
        def f(node):
            if not node.children:
                # If we had a tree with unique string identifiers, we could use return 1, 0
                # 0 is here because h[leaf] = 0, since a leaf won't be a LCA
                C[node] = len(node.string_ids)
                return len(node.string_ids), 0

            S = 0
            U = 0
            for child in node.children.values():
                S_c, U_c = f(child)
                S += S_c
                U += U_c

            U += h[node]
            C[node] = S - U
            return S, U

        f(self._root)
        
        # Step 8
        string_depth = {}
        def f(node, depth):
            string_depth[node] = depth
            for child in node.children.values():
                if child.children:
                    f(child, depth + child.edge_length())
                else:
                    # This stops us from counting the terminal character as part of a shared substring
                    f(child, depth + child.edge_length() - 1)

        f(self._root, 0)

        V = [[0, None]] * (len(self._strings) + 1) # Using 1-indexing here (unused V[0] value)
        for v, k in C.items():
            v_depth = string_depth[v]
            if v_depth > V[k][0]:
                V[k] = [v_depth, v]
            # Deals with the case when two strings have the same length (excluding the terminal character, which will have depth 0)
            elif v_depth == V[k][0] and v_depth != 0:
                V[k].append(v)

        for k in range(len(V) - 2, 0, -1):
            if V[k][0] < V[k + 1][0]:
                V[k] = V[k + 1]

        # Here we are only using the last value of V(k) (which is l(k) now) for the LCS of all the strings, but
        #  the above code is useful if the full array l(k) is desired
        lcs_nodes = V[-1][1:]
        if lcs_nodes[0] == None:
            return None
        
        lcs_strings = []
        for lcs_node in lcs_nodes:
            lcs_string = self._node_string(lcs_node)
            lcs_strings.append(lcs_string)
        return list(set(lcs_strings))
    

    def _node_string(self, node):
        # Given a node, return the string that node represents
        reversed_path = []
        while node.parent is not None:
            reversed_path.append(node)
            node = node.parent

        path = reversed_path[::-1]

        string = ''
        for node in path:
            string += self._strings[node.string_id][node.start:node.end + 1]

        if string[-1] == self._terminal_character:
            return string[:-1]
        return string
    

    def _create_graph(self, graph, suffix_link, node):
##        dfs_num = str(node.dfs_num)
        # Leaves
        if not node.children:
            graph.node(node.num, f'{node.dfs_num}: ' + ', '.join([f'({i}, {j})' for i, j in zip(node.string_ids, node.start_idxs)]), shape='rect', fontsize='10')
##            graph.node(node.num, str(node.dfs_num) + ': ' + ', '.join(str(i) for i in node.string_ids), shape='circle', fontsize='10')
##            graph.node(node.num, str(node.dfs_num) + ", " + bin(node.dfs_num)[2:])
##            graph.node(node.num, str(node.dfs_num))
            
        # Internal nodes
        if node.children:
##            graph.node(node.num, '')
##            graph.node(node.num, str(node.dfs_num) + ", " + bin(node.dfs_num)[2:])
            graph.node(node.num, str(node.dfs_num))
            # SL Links
            if suffix_link and node.suffix_link:
                graph.edge(node.num, node.suffix_link.num, style='dotted')

            for child in node.children.values():
                # Edges
                graph.edge(node.num, child.num, label=self._strings[child.string_id][child.start:child.end + 1])
                self._create_graph(graph, suffix_link, child)
                

    def create_graph(self, suffix_link=False):
        import os
        # The os.environ['PATH'] variable is diffferent sometimes depending on how Python is instantiated
        # Works fine from terminal
        os.environ["PATH"] += os.pathsep + '/usr/local/bin'
        from graphviz import Digraph
        
        graph = Digraph(graph_attr={'rankdir': 'LR'},
                        node_attr={'shape': 'circle', 'height': '0.1', 'weight': '0.1'},
                        edge_attr={'arrowsize': '0.4', 'fontsize': '10', 'weight': '3'})
        self._create_graph(graph, suffix_link, self._root)
        return graph
            
    

class LCA:
    def __init__(self, root):
        self._root = root
        self._number_nodes()
        self._compute_I_and_L()
        self._compute_A()
        

    @staticmethod
    def h(x):
        # Same as lsb
        # Using https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
        #  Other methods may overall be faster in large computations (check out builtin ffs function or lookup table)
        return (x & -x).bit_length()


    @staticmethod
    def msb(x):
        return x.bit_length()


    def _number_nodes(self):
        count = 1

        def dfs_num(node):
            nonlocal count
            node.dfs_num = count
            count += 1

            for child in node.children.values():
                child.parent = node
                dfs_num(child)

        return dfs_num(self._root)


    # Based on strmat library.  This is from stree_lca.c
    def _compute_I_and_L(self):
        I = {}
        L = {}
        
        def f(node):
            # TODO: We may be able to speed this up by storing h values of nodes on the node
            #  and using an h_Ival variable
            Imax = node.dfs_num   # TODO: ENSURE NODE NUMBERING IS DONE BEFOREHAND
            for child in node.children.values():
                Ival = f(child)
                if self.h(Ival) > self.h(Imax):
                    Imax = Ival

            I[node] = Imax
            L[Imax] = node # Will be overwritten by the highest node in run
            # TODO: I believe the above line will take up useless space in the dictionary

            return Imax

        f(self._root)
        self._I = I
        self._L = L


    def _compute_A(self):
        A = {}
        
        def f(node, A_mask):
            A_mask |= 1 << (self.h(self._I[node]) - 1)  # TODO: If we store h on the node then we don't have to recompute
            A[node] = A_mask
            for child in node.children.values():
                f(child, A_mask)

        f(self._root, 0)
        self._A = A


    def lca(self, x, y):
        # The book algorithm only handles the case that lca(x, y) != x or y
        # This is for the case that lca(x, y) = x or y and is from the original paper:
        #  Finding Lowest Common Ancestors Simplication and Parallelization
        if self._I[x] == self._I[y]:
            if x.dfs_num <= y.dfs_num:
                return x
            return y

        # The text (top of page 192) seems to imply that we can calculate i by using i = h(I[x] & I[y]) (the right-most common 1-bit),
        #  but this method doesn't seem to work, and is not implemented in strmat
        # I believe this is a typo, and the text means the left-most differing 1-bit, which would make sense,
        #  as the first differing left bit is the amount of nodes on their paths from root which they DON'T share
        # Furthermore, i is found in the original paper using
        #  i = floor(log2(I[x] ^ I[y])) + 1, which is equivalent to i = msb(I[x] ^ I[y]) (with msb/i being 1-indexed)

        # Step 1
        i = self.msb(self._I[x] ^ self._I[y])

        # strmat version
    ##    k = msb(I[x] ^ I[y]) - 1
    ##    mask = ~0 << (k + 1)
    ##    b = (I[x] & mask) | (1 << k)
    ##    i = h(b)

        # Step 2
        mask = ~0 << (i - 1)
        j = self.h(self._A[x] & self._A[y] & mask)

        # Step 3
        l = self.h(self._A[x])
        if l == j:
            x_bar = x
        else:
            mask = ~(~0 << (j - 1))
            k = self.msb(self._A[x] & mask)

            mask = ~0 << k
            I_w = (self._I[x] & mask) | (1 << (k - 1))
            w = self._L[I_w]
            x_bar = w.parent

        # Step 4
        l = self.h(self._A[y])
        if l == j:
            y_bar = y
        else:
            mask = ~(~0 << (j - 1))
            k = self.msb(self._A[y] & mask)

            mask = ~0 << k
            I_w = (self._I[y] & mask) | (1 << (k - 1))
            w = self._L[I_w]
            y_bar = w.parent

        # Step 5
        if x_bar.dfs_num <= y_bar.dfs_num:
            return x_bar
        return y_bar
    


##test = SuffixTree(['abcabxabcd'])
##test = SuffixTree(['abcab'])
##test = SuffixTree(['abcdababe'])
##test = SuffixTree(['abcabxabcd', 'abcdababe'])
##test = SuffixTree(['xabxa'])
##test = SuffixTree(['xabxa', 'babxba'])
##test = SuffixTree(['abc', 'bc'])
##test = SuffixTree(['aaaacbbbedddd', 'aaaadbbbfddddc', 'c'])
##test = SuffixTree(['GATTACA', 'TAGACCA', 'ATACA'])
##test = SuffixTree(['aaacaaa', 'aaadaaa'])
##test = SuffixTree(['abcdef', 'ghijk'])
##test = SuffixTree(['sandollar', 'sandlot', 'handler', 'grand', 'pantry'])
##test = SuffixTree(['abcdefabxybcdmnabcdex'])
##test = SuffixTree(['forgeeksskeegfor', 'forgeeksskeegfor'[::-1]])
##test = SuffixTree(['babad', 'babad'[::-1]])
##test = SuffixTree(['cbbd', 'cbbd'[::-1]])

##graph = test.create_graph(False)
##graph.render('/tmp/test.gv', view=True)
##print(test.find('ab'))
##print(test.lcs())



# This is an attempted (**incomplete**) commentary/proof that this algorithm correctly implements the Single Phase Algorithm
#  Not all statements below are true (and some may be contradictory).  These statements helped me write parts of this code and may help someone else for the time being

##def _add_char(self, c):
##    # This is the SPA algorithm from Dan Gusfield's book.  We use an explanation/implementation heavily referenced from
##    # https://stackoverflow.com/questions/9452701/ukkonens-suffix-tree-algorithm-in-plain-english
##    # https://gist.github.com/makagonov/22ab3675e3fc0031314e8535ffcbee2c
##    # https://www.geeksforgeeks.org/ukkonens-suffix-tree-construction-part-6/
##    # Helpful visualization: http://brenden.github.io/ukkonen-animation/
##
##    # On the bottom of page 105, it is inferred that Rule 1 can occur again after adding S(i+1) to every leaf
##    #  But after adding S(i + 1) to every leaf, beta cannot end on a leaf (it ends on an edge from the prior node, since every leaf has added S(i + 1)),
##    #   so Rule 1 can never occur afterwards in this phase
##    #  Since we stop at Rule 3, only Rule 2 can occur inbetween Rule 1 and 3
##    #  Thus the rules are performed in order -- Rule 1, Rule 2, Rule 3
##
##    # Suffix Links
##    #  Suffix links are created in SEA 4 only when Rule 2 with an edge has occured
##    #  First we perform Rule 1, where no suffix links are created
##    #  We then perform a string of Rule 2's.  After the first Rule 2, we must check if the prior Rule 2 created an internal node to add a suffix link to
##    #  We then perform Rule 3, and must again check if the prior Rule 2 (if it exists) added a new internal node
##    #  
##
##
##    # TODO: I BELIEVE WE ACTUALLY START EVERY PHASE WITH beta = S[j*..i]
##    # TODO: NOT SURE IF ALWAYS END/START WITH S[j*..i-1].. LOOK AT EXTENSION RULE 3
##    # We enter phase i + 1 having ended with beta = S[j*..i-1] in the last phase (IS THIS WHERE ACTIVE_POINT POINTS !?)
##    #  where j* is either the extension prior to Rule 3 in the last phase, the last iteration of phase
##    #  i (i = j*, note we can have S[i..i-1] for a blank suffix ''), or 1 for the first phase
##    # TODO: IS j* LAST PRIOR TO RULE 3 OR RULE 3 !??!  I THOUGHT WE THOUGHT IT WAS RULE 3 ITSELF
##    
##    # SPA 1. / SEA 1. adds all suffixes S[1 to j*..i+1] to the suffix tree by Trick 2 (iterations 1 through j*)
##    SuffixTreeNode.leaf_end = self._phase
##
##    # At this point we are at iteration j* with active_node pointing at S[j*..i-1] <-- VERIFY THIS
##    
##    # SPA 2./SPA 3. (represented by remainder -=1) Here remainder represents extensions > j_{i+1}
##    self._need_suffix_link = None
##    self._remainder += 1
##    while self._remainder > 0:
##        if self._active_length == 0:
##            self._active_edge_index = self._phase
##            self._active_edge = self._string[self._active_edge_index]
##
##        # At this point active_point = (active_node, active_edge, active_length) 'points' at beta = S[j..i] OR
##        #  we will find beta after walking down (one or more times)
##        
##        next_node = self._active_node.children.get(self._active_edge)
##        if next_node:
##            # SEA 2. using the Skip/Count Trick to walk down.  If we walk down we restart the loop and keep walking down until beta has been found
##            if self._walk_down(next_node):
####                    print("Walk down")
##                continue 
##            
##            # Extension Rule 3.  We break according to SPA Step 2. / Observation 1 (Rule 3 is a show stopper)
##            elif self._strings[next_node.string_id][next_node.start + self._active_length] == c:
####                if self._string[next_node.start + self._active_length] == c:
##                # active_point = (active_node, active_edge, active_length) = beta = S[j..i]
##                # We still are on the same edge of active_node, but the next beta is one position further on that edge so
##                #  we have active_point = (active_node, active_edge, active_length + 1) = beta = S[j..i+1]
##                # Note: We should NOT have to walk down after this operation
##                #  since we have already walked down above to ensure this operation is valid
##                # TODO: Ex: S[j..i] = abc
##                if c == self._terminal_character:
####                        print("ER3 terminal")
##                    next_node.string_ids.append(self._string_id)
##                    # When performing ER3, we may have to add in a final suffix link
##                    # When using the ER3 with the terminal character, ER3 may be called multiple times
##                    #  rather than once as it usually is (hence the break in that block of code)
##                    # We use self._terminal_er3 to make sure we only add in the final suffix link and don't overwrite others
##                    #  (as this causes bugs, such as with ['aaacbbb', 'aaadbbb'])
##                    if not self._terminal_er3:
##                        self._add_suffix_link(self._active_node)
##                        self._terminal_er3 = True
##                else:
####                        print("ER3")
##                    self._active_length += 1
##                    self._add_suffix_link(self._active_node)
##                    break
##            else:
##                # Extension Rule 2. for beta ending inside an edge
##                # TODO: The below sentence is unclear
##                # After ER2, the only node with a suffix link will be the current active_node.  So we use this node's suffix link.  This node may also be root.
##                #  The new split_node will be the new internal node, and will receive it's suffix link next extension by Corollary 6.1.1
####                    print("ER2 edge")
##                split_node = SuffixTreeNode(next_node.start, next_node.start + self._active_length - 1, next_node.string_id)
##                self._active_node.children[self._active_edge] = split_node
##                leaf = self._add_leaf()
##                split_node.children[c] = leaf
##                next_node.start += self._active_length
##                split_node.children[self._strings[next_node.string_id][next_node.start]] = next_node
##                self._add_suffix_link(split_node)
##        else:
####                print("ER2 leaf")
##            # TODO: VERIFY BELOW
##            # By reaching this point, we have ensured active_point points at beta (since we have walked down) and that we
##            #  have ended on an internal node, since we can't end on a leaf and there was no edge out of this node.
##            # Since beta ended on an internal node, we have active_length == 0 -> active_edge = c by the first if statement.             
##            # Thus we have ended on an internal node (where internal node here includes root) that has a labeled path that continues (since it is an internal node)
##            #  but is not equal to our character c = S(i + 1) OR we are on the first phase with the first insertion of a single character.
##            # Thus by Extension Rule 2 a new leaf must be created with character c (no internal node is created)
##
##            # Extension Rule 2 for beta not ending at an edge (instead landing at an internal node)
##            leaf = self._add_leaf()
##            self._active_node.children[self._active_edge] = leaf
##            self._add_suffix_link(self._active_node)
##
##        # Note that since we have reached this point, we have not performed Extension Rule 3 (or 1) and so we have
##        #  performed Extension Rule 2.  Thus we have inserted a new node which is a child of the active node
##        # OR we have done ER3 on a terminal character if we are using a GST
##        
##        # SEA 1./part of SEA 2. (traversing the suffix link, walking down is done above) of the next extension (j + 1)
##        if self._active_node == self._root and self._active_length > 0:
##            # We have inserted a new node (either leaf or internal node) as a child of root.  There is no suffix link to follow.
##            #  active_point = beta = S[j..i] = (root, active_edge, active_length)
##            #  By increasing active_edge by 1 (which means active_length must be reduced by one) we move to S[j+1..i], or
##            #  we can walk down to it
##            # Ex: S[j..i] = abcde
##            # active_point = (root, 'a', 5)
##            #  active_point will 'point' at beta = S[j..i] = abcde.  Note active_node = root = ''
##            #  By increasing active_edge_index by 1 and decreasing active_lenth by 1, we move to (root, 'b', 4),
##            #  At this point active_point now points at S[j+1..i] or an ancestor of it, S[j+1..i-k],
##            #  which we can walk down from to get to S[j+1..i]
##
##            self._active_edge_index += 1
##            self._active_edge = self._string[self._active_edge_index]
##            self._active_length -= 1
##        elif self._active_node != self._root:
##            # Half of SEA 2. with v not root (traverse the suffix link, not walk down)
##            
##            # We have just performed Extension Rule 2
##            # We go from the active_point, S[j..i], to some ancestor of it (S[j..i-k]), the active_node, and follow the suffix link there.
##            #  This leaves us at some ancestor of S[j+1..i], which we may immediately arrive at (I believe this is always the case when using Extension Rule 2 to add a leaf)
##            #  or may need to walk down to
##            # Ex: S[j..i] = xabcdef
##            #  active_point will 'point' at beta = S[j..i] = xabcdef.  Say active_node is at xabc
##            #  Then taking the suffix link from active_node, we arrive at a node labeled abc
##            #  We walk down from abc to get to abcdef = S[j+1..i]
##
##            # By following the suffix link, we go from S[j*-1..i] to S[j*..i], where j* is the last Extension Rule 3 or the end of the phase (j* == i + 1)
####                print(self._active_node.start, self._active_node.end, self._active_node == self._root)
##            self._active_node = self._active_node.suffix_link
##            
##        self._remainder -= 1
##
##        # If we end a phase here:
##        # Because we did not end on Rule 3 (otherwise we would have broken out by now), we have done every iteration
##        #  and are ending on iteration j=i+1 with beta = S[i+1..i] = ''.  (Here we are just inserting a leaf with S(i+1) on it)
##        # This means beta = '', which means active_point is pointing at root
##        #  with active_point = (root, some letter, 0)    (some letter could be null in another implementation)
##        # TODO: I'M NOT SURE OF THE BELOW AND WHAT BETA BECOMES EXACTLY
##        # TODO: TRY TO DESCRIBE WHERE WE END THE PHASE HERE
##        #  DESCRIBE HOW WE GET TO THE PROPER BETA FOR THE NEXT PHASE AT THE TOP, WHERE THE NEW PHASE STARTS
##        #  SO KEEP IN MIND WHAT HAPPENS HERE WHEN WE END THE PHASE (TEXT ABOVE) AND TRY TO DROP TEXT BELOW OR MOVE IT UP
##        # So in the next phase active_node will be 
##        # In the next phase, after performing SPA 1. (Extension Rule 1's), the first if statement (active_length == 0)
##        #  will set active_point = (root, S(i + 2), 0).
##        # Since active_node = root and active_length is still 0, this implies beta is still '' = S[i+2..i+1].
##        # We set active_edge to S(i + 2) to ensure the i + 3 phase is properly set <---- TODO: CONFIRM THIS
##        # I BELIEVE WE CAN ONLY GO TO ER3 OR ER2 WITH LEAF FROM HERE (NOT ER2 WITH INTERNAL NODE) IN PHASE i + 2 I MEAN
##        
##    self._phase += 1
