import os
import math
import json
import unittest
import numpy as np
from tqdm import tqdm

from textrank.textrank import TextRank

def intersect( A, B ):

    return set( A ) & set( B )

def precision( assigned, correct ):
    
    intersections_lens = [ len( intersect( assigned[ i ], correct[ i ] ) ) for i in range( len( assigned ) ) ]
    
    assigned_lens = [ len( item ) for item in assigned ]
    print ( f'average matches={np.mean( intersections_lens )} out of {np.mean( assigned_lens)}')
     
    return np.sum( intersections_lens ) / np.sum( assigned_lens )


def recall( assigned, correct ):

    intersections_lens = [ len( intersect( assigned[ i ], correct[ i ] ) ) for i in range( len( assigned ) ) ]
    correct_lens = [ len( item ) for item in correct ]
     
    return np.sum( intersections_lens ) / np.sum( correct_lens )

def f1( assigned, correct ):

    p = precision( assigned, correct )
    r = recall( assigned, correct )

    return 2. * ( p * r ) / ( p + r )


class TestTextRank( unittest.TestCase ):
    def setUp( self ):
        self.textrank  = TextRank( N=2 )
        self.assetpath = os.path.join( os.getcwd(), 'tests/assets' )
        self.text = 'Low index linear systems with those properly stated leading terms are considered in detail. In particular, it is asked whether a numerical integration method applied to the original system reaches the inherent regular ODE without conservation, i.e., whether the discretization and the decoupling commute in some sense. In general one cannot expect this commutativity so that additional difficulties like strong stepsize restrictions may arise. Moreover, abstract differential algebraic equations in infinite-dimensional Hilbert spaces are introduced, and the index notion is generalized to those equations. In particular, partial differential algebraic equations are considered in this abstract formulation'
        
        # load the inspect test set json
        with open( os.path.join( self.assetpath, 'test.uncontr.json' ) ) as f:
            self.testset = json.load( f )

    def testTextRankOutputAndScores( self ):
        ranking = self.textrank.rank( self.text )

        [ self.assertIsInstance( pair, tuple ) for pair in ranking ]

        score = math.inf

        # check that the results are sorted in descending order
        for k, v in ranking:
            self.assertTrue( v <= score )
            score = v

    def evaluateInspecDataset( self, N ):
        # the ground truth - the batch of true labels 
        # for each test example
        tl = []
        # the model's labels 
        hl = []

        assigned_count = []
        correct_count = []

        self.textrank = TextRank( N=N )

        print( f'evaluating textrank with INSPECT testset N={N}' )

        # load each test document in turn
        for idx in tqdm( list( self.testset.keys() ) ):
            file = os.path.join( self.assetpath, f'{idx}.txt' )
            text = ''

            with open( file ) as f:
                text = f.read().replace( '\n', ' ' )

            # perform keyword extraction
            keywords = self.textrank.multiword_keywords( text )

            new_hl = keywords
            new_tl = [ label[0] for label in self.testset[ idx ] ]

            inters = set( new_hl ) & set( new_tl ) 
            # print ( sorted( new_hl ) )
            # print ( sorted( new_tl ) )
            # print ( inters )
            # print ( f'matches: { len(inters)} out of {len(new_tl)}' )
            # input( 'press key to continue' )
            hl.append( new_hl )
            tl.append( new_tl )
            assigned_count.append( len( new_hl ) )
            correct_count.append( len( inters ) ) 


        print ( f'assigned: total={np.sum( assigned_count )}, mean={np.mean( assigned_count )}' )
        print ( f'correct:  total={np.sum( correct_count)}, mean={np.mean( correct_count )}' )
        print ( f'precision={precision( hl, tl )}, recall={recall( hl, tl )}, f1={f1( hl, tl )}' )
    

    def testTextRankInspecDataset( self ):

        windows = [ 1, 2, 3, 4, 5, 10 ]

        for N in windows:
            self.evaluateInspecDataset( N )


