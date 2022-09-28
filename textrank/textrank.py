from itertools import combinations
import math 

import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

import matplotlib.pyplot as plt

import networkx as nx

class TextRank( object ):
	"""
		It implements the Textrank algorithm from:

			“TextRank: Bringing Order into Texts"
			R. Mihalcea, P. Tarau, 2004

		Parameters:
		-----------
		
			N - the size of the window, in lexical units (e.g. words, sentences) that defines 
				the co-occurrence relation. Default is 2 logical lexical units.

			pos - an array of syntactic filters, codes for the parts of speech that are to be considered 
				  nodes of the graph (e.g. only use nouns and verbs). Default is noun and adjectives.
			
			T - the number of top ranked lexical units to return. Defaults to a third of the vertices in the graph

	"""
	def __init__( self, N=2, pos=[ 'NN', 'JJ' ] ):
		
		self._N = N 
		self._pos = pos

	def preprocess( self, text ):

		pos = self._pos

		# first, the text is tokenised and annotated with parts of speech tags
		tokens = word_tokenize( text.lower() )
		tokens = [ token for token in tokens if token not in get_stop_words( 'en' ) ]

		annotated_tokens = nltk.pos_tag( tokens )

		# gather the lexical units that pass the filter(s)
		vertices = list ( set ( [ token for token, annotation in annotated_tokens if any( [ annotation.startswith( prefix ) for prefix in pos ] ) ]  ) )
		vertices = sorted( vertices )

		return tokens, vertices

	def rank( self, text, tol=0.0001 ):

		tokens, units = self.preprocess( text )

		self._lenunits = len( units )
		
		nxgraph = self.graph( units, tokens )

		scores = nx.pagerank( nxgraph, tol=tol )

		unsorted_ranking = [ ( units[ vertex ], score ) for vertex, score in scores.items() ]
		
		return sorted( unsorted_ranking, key=lambda x: x[ 1 ], reverse=True )


	def graph( self, vertices, tokens, plot=False ):

		N = self._N
		lentokens = len( tokens )

		idx2vertex = { idx:vertex for idx, vertex in enumerate( vertices ) }
		vertex2idx = { vertex:idx for idx, vertex in enumerate( vertices ) }

		graph = np.zeros( [ len( vertices ) ] * 2 )

		for idx in range( lentokens ):
			for x, y in list( combinations( tokens[ max( 0, idx-N ): min( lentokens-1, idx+N+1 ) ], 2 ) ):
				try:
					v1 = vertex2idx[ x ]
					v2 = vertex2idx[ y ]

					if v1 != v2:
						graph[ v1, v2 ] = 1.
						graph[ v2, v1 ] = 1.
				except:
					continue

		nxgraph = nx.from_numpy_array( graph )

		if plot:
			H = nx.relabel_nodes( nxgraph, idx2vertex )
			nx.draw_networkx( H, with_labels=True )

		return nxgraph


	def keywords( self, text, T=None ):

		ranking = self.rank( text )
		words = [ token for token, score in ranking ]

		if T is None:
			T = math.floor( self._lenunits / 3 )

		return words[ :T ]

	def _append( self, tokens, keywords, target, curridx ):

		target = target.strip() 

		if curridx < len( tokens ):
			if tokens[ curridx ] in keywords:
				target = f'{target} {tokens[ curridx ]}'
				return self._append( tokens, keywords, target, curridx + 1 )

		return target, curridx + 1

	def multikeywords( self, text, T=None ):

		ranking = { token:score for ( token, score ) in self.rank( text ) }
		keywords = list( ranking.keys() )

		tokens = word_tokenize( text.lower() )
		multiwords = []

		curridx = 0

		while curridx < len( tokens ):
			multiword, curridx = self._append( tokens, keywords, '', curridx ) 
			if multiword:
				multiwords.append( multiword )

		multiwords = np.array( list( set ( multiwords ) ) )

		# score and rank multikeywords - the score of a multikeyword is the
		# sum of the scores if its components keywords
		multiranking = [ sum( [ ranking[ token ] for token in k.split( ' ' ) ] ) for k in multiwords ]

		if T is None:
			T = math.floor( self._lenunits / 3 )

		return multiwords[ np.argsort( multiranking ) ][ ::-1 ][ :T ]

