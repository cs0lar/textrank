import numpy as np


def graphrank( G, d=0.85, sinit=1., threshold=0.0001 ):

	rows, cols = G.shape
	scores = np.array( [ sinit ]*rows )
	new_scores = np.zeros( rows )
	done = False

	while not done:
		for i in range( rows ):

			incoming = np.where( G[ :, i ] > 0 )[ 0 ]

			outgoings = np.sum( G[ incoming, : ], axis=1 )

			new_scores[ i ] = ( 1. - d ) + d * np.sum( ( 1. / outgoings ) * scores[ incoming ] )

		diff = np.abs( new_scores - scores )
		scores = new_scores
		done = np.all( diff<=threshold )

	return { i:score for i, score in enumerate( scores ) }



def weightedgraphrank( G, W, d=0.85, sinit=1., threshold=0.0001 ):

	if G.shape != W.shape:
		raise ValueError( 'Graph G and weights matrix W must have the same shape' )

	rows, cols = G.shape
	scores = np.array( [ sinit ]*rows )
	new_scores = np.zeros( rows )
	done = False

	while not done:
		for i in range( rows ):

			incoming = np.where( G[ :, i ] > 0 )[ 0 ]

			outgoings = [ np.nonzero( row ) for row in G[ incoming, : ] ]

			w = [ W[ incoming[ j ], i ] / np.sum( W[ incoming[ j ], out ] ) for j, out in enumerate( outgoings ) ]

			new_scores[ i ] = ( 1. - d ) + d * np.sum( w * scores[ incoming ] )

		diff = np.abs( new_scores - scores )
		scores = new_scores

		done = np.all( diff>threshold )


	return { i:score for i, score in enumerate( scores ) } 







