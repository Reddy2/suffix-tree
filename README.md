# suffix-tree
Generalized suffix trees in Python built in linear time using Ukkonen's Algorithm.  Implements the linear time LCA preprocessing algorithm with constant time retrieval.  Supports a few applications of generalized suffix trees, as well as the ability to graph the tree using graphviz.

## Construction
```Python
>>> tree = SuffixTree(['GATTACA', 'TAGACCA', 'ATACATA'])
```

## Implemented Algorithms
### Find all occurrences of a substring
```Python
>>> tree = SuffixTree(['GATTACA', 'TAGACCA', 'ATACATA'])
>>> tree.find('TA')
[(1, 0), (2, 1), (0, 3), (2, 5)]
```
The numbering here is (string_id, starting_index).  For example, the third string (string_id = 2) has the substring 'TA' start at indexes 1 and 5.  Note that the results are **not** sorted.

### Longest Common Substring
```Python
>>> tree = SuffixTree(['GATTACA', 'TAGACCA', 'ATACATA'])
>>> tree.lcs()
['CA', 'AC', 'TA']
```

## Graphing
```Python
>>> graph = tree.create_graph(suffix_link=False)  # Same as tree.create_graph()
>>> graph.render('/tmp/test.gv', view=True) # graph is a graphviz object
```
Creates a graph of the suffix tree using graphviz, with or without suffix links.  The internal nodes are labeled with their 'depth first search' (dfs) numbers.  The leaves are labeled as follows: dfs: [(string_id_0, starting_index_0), (string_id_1, starting_index_1), ...] where string_id is the index of the string in the array given to the tree upon construction.  For example, in SuffixTree(['GATTACA', 'TAGACCA', 'ATACATA']), 'TAGACCA' would have string_id = 1.  
Note that graphviz is a Python module but you must also download the graphviz software.  I had a bit of trouble using graphviz in IDLE due to PATH issues (I added a line of code to hopefully fix this for others) but found it to work fine in terminal (on Mac).

## Notes
- Currently the dollar sign $ is used as a terminal character.  You should change self._terminal_character if you are using strings with the $ sign for now. 
- The generalized suffix tree will use the same terminal character for every string.  Some algorithms are typically described by using a different terminal character for each string, so keep this in mind if you are trying to learn from this code/add an algorithm.
- This code is tested using problems from [Rosalind](http://rosalind.info) but is not verified to be correct.  If you are looking for speed, then you should use a different library.  In addition to using a faster language, you may want to look for a suffix array alternative of an algorithm or use one of the distributed construction algorithms for suffix trees.
- The LCA preprocessing algorithm needs some tuning (see Gusfield section 8.10. For the purists: how to avoid bit-level operations) to be truly linear and will probably run into stack overflow issues with too large a dataset due to recursion being used for DFS.

## Helpful resources
I used these resources to help implement this
- Ukkonen's original paper: https://www.cs.helsinki.fi/u/ukkonen/SuffixT1withFigs.pdf
- Algorithms on Strings, Trees, and Sequences: Computer Science and Computational Biology (Dan Gusfield)
- https://stackoverflow.com/questions/9452701/ukkonens-suffix-tree-algorithm-in-plain-english and the code provided from the answer https://stackoverflow.com/a/14580102
- https://www.geeksforgeeks.org/ukkonens-suffix-tree-construction-part-1/ (helpful hints relating Gusfield to the stackoverflow answer)

I have attempted to provide a (messy, incomplete) proof of my algorithm's correctness related to Gusfield's algorithm for those interested.  It is at the bottom of the file and helped me implement the algorithm.  I will attempt to add more suffix tree applications as I complete the Rosalind problems.
