# text-search
Creates inverted index for a big text collection and return ranked lists of relevant documents for search queries.

This code should be executed from the command line as:
                        python3 SearchIndex.y filename rank_method num_results query a_max u
                        
-The file should contain all the text in the text-collection that we want to index and search. 
-The rank method could be either "proximity" or "bm25".
-num_results is the maximum number of results we want to be displayed
-a_max (maximum size ofaccumulators for accumulator pruning) and u (the threshold for bm25) are parameters to the bm25 implementation.
 This is given as a command-line argument instead of harcoding to provide the user with flexibilty to tailor the bm25 
 implemetation to suit their requirements easily.
