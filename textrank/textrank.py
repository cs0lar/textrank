from itertools import combinations

import nltk
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

from .graphrank import graphrank
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
	def __init__( self, N=2, pos=[ 'NN', 'JJ' ], T=None ):
		
		self._tokeniser = RegexpTokenizer( r"([\w]+(?:(?!\s)\W?[\w]+)*)" )
		self._N = N 
		self._pos = pos
		self._T = T 

	def rank( self, text ):

		N = self._N
		pos = self._pos
		T = self._T

		# first, the text is tokenised and annotated with parts of speech tags
		tokens = self._tokeniser.tokenize( text.lower() )
		tokens = [ token for token in tokens if token not in get_stop_words( 'en' ) ]
		annotated_tokens = nltk.pos_tag( tokens )

		# gather the lexical units that pass the filter(s)
		units = list ( set ( [ token for token, annotation in annotated_tokens if any( [ annotation.startswith( prefix ) for prefix in pos ] ) ]  ) )
		units = sorted( units )
		unit2vertex = { v:i for i, v  in enumerate( units ) }

		# create the graph of lexical units
		graph = np.zeros( ( len( units ), len( units ) ) )

		# and add an (undirected) edge between those lexical units that co-occur within a window of N units
		for i in range( len( tokens ) - N + 1 ):
			for x, y in list( combinations( tokens[ i:i+N ], 2 ) ):
				try:
					v1 = unit2vertex[ x ]
					v2 = unit2vertex[ y ]
					if v1 != v2:
						graph[ v1, v2 ] = 1.
						graph[ v2, v1 ] = 1.
				except:
					continue

		# run graphrank on it until it converges
		nx_graph = nx.from_numpy_array( graph )
		scores = nx.pagerank( nx_graph )

		# sort the vertices in reverse order and retain the top T
		if T is None:
			T = int( len( units ) * 0.67 )

		unsorted_ranking = [ ( units[ vertex ], score ) for vertex, score in scores.items() ]
		ranking = sorted( unsorted_ranking, key=lambda x: x[ 1 ], reverse=True )

		return ranking[ :min( len( units ), T ) ]


	def keywords( self, text ):

		ranking = self.rank( text )

		return [ token for token, score in ranking ]

	def _append( self, tokens, keywords, target, curridx ):

		target = target.strip() 

		if curridx < len( tokens ):
			if tokens[ curridx ] in keywords:
				target = f'{target} {tokens[ curridx ]}'
				return self._append( tokens, keywords, target, curridx + 1 )

		return target, curridx + 1

	def multiword_keywords( self, text ):

		keywords = self.keywords( text )

		tokens = word_tokenize( text.lower() )
		multiwords = []

		curridx = 0

		while curridx < len( tokens ):
			multiword, curridx = self._append( tokens, keywords, '', curridx ) 
			if multiword:
				multiwords.append( multiword )

		return list( set( multiwords ) )

		



if __name__ == '__main__':
	textrank = TextRank( N=2 )

	# keywords = textrank.keywords( 'Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict in-equations, and non-strict in-equations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types.')

	# print ( keywords )
	multiwords = textrank.multiword_keywords( 'Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict in-equations, and non-strict in-equations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types.')

	print( multiwords )