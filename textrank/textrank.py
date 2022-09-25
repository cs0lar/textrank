from itertools import combinations

import nltk
import numpy as np

from nltk.tokenize import RegexpTokenizer
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
	def __init__( self, N=2, pos=[ 'NN', 'JJ', 'VBP' ], T=None ):
		
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
		unit2vertex = { v:i for i, v in enumerate( units ) }
		vertex2unit = { i:v for i, v in enumerate( units ) }

		# create the graph of lexical units
		graph = np.zeros( ( len( units ), len( units ) ) )

		# lentokens = len( tokens )

		# # and add an (undirected) edge between those lexical units that co-occur within a window of N units
		# for i in range( lentokens ):
		# 	# print ( f'processing {tokens[ max( 0, i-N ): min( lentokens-1, i+N+1 ) ]}')
		# 	for x, y in list( combinations( tokens[ max( 0, i-N ): min( lentokens-1, i+N+1 ) ], 2 ) ):
		# 		try:
		# 			v1 = unit2vertex[ x ]
		# 			v2 = unit2vertex[ y ]
		# 			if v1 != v2:
		# 				graph[ v1, v2 ] = 1.
		# 				graph[ v2, v1 ] = 1.
		# 		except:
		# 			continue
		# print (annotated_tokens)
		# print ( units )
		d = {
			'systems': [ 'types', 'compatibility', 'linear' ],
			'compatibility': [ 'systems', 'criteria' ],
			'criteria': ['compatibility', 'natural', 'numbers'],
			'natural': ['criteria', 'numbers'],
			'numbers': ['criteria', 'natural'],
			'types': ['systems', 'solutions'],
			'linear': ['systems', 'system', 'constraints', 'diophantine', 'equations'],
			'system':['linear'],
			'constraints':['linear'],
			'equations':['linear', 'strict'],
			'diophantine':['linear'],
			'strict':['equations', 'inequations'],
			'inequations':['strict', 'nonstrict'],
			'solutions':['types', 'algorithms', 'sets'],
			'algorithms':['solutions', 'construction'],
			'sets':['solutions', 'minimal'],
			'minimal':['sets', 'construction', 'components'],
			'construction':['algorithms', 'minimal'],
			'components':['minimal', 'bounds'],
			'bounds':['components', 'upper'],
			'upper':['bounds'],
			'nonstrict':['inequations']
		}

		for k in d.keys():
			for w in d[ k ]:
				v1 = unit2vertex[ k ]
				v2 = unit2vertex[ w ]

				graph[v1,v2] = 1


		# run graphrank on it until it converges
		nx_graph = nx.from_numpy_array( graph )
		scores = nx.pagerank( nx_graph,tol=0.0001 )
		# H = nx.relabel_nodes( nx_graph, vertex2unit )
		# nx.draw_networkx( H, with_labels=True )

		# sort the vertices in reverse order and retain the top T
		if T is None:
			T = len(units)#int( len( units ) / 3 )

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

		multiwords = list( set ( multiwords ) )

		# return multiwords
		result = []
		for i in range( len( multiwords ) ):
			skip = False
			for j in range( len( result ) ):
				a = multiwords[ i ]
				b = result[ j ]
				pair = ( a, b )

				if a in b or b in a:
				
					result[ j ] = pair[ np.argmax( ( len( a ), len( b ) ) ) ]
					skip = True

			if not skip:
				result.append( multiwords[ i ] )
			result = list( set ( result ) )

		return result



if __name__ == '__main__':
	textrank = TextRank( N=3 )
	text = 'Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types.'
	# text = 'Mathematical fundamentals of constructing fuzzy Bayesian inference techniques Problems and an associated technique for developing a Bayesian approach to decision-making in the case of fuzzy data are presented. The concept of fuzzy and pseudofuzzy quantities is introduced and main operations with pseudofuzzy quantities are considered. The basic relationships and the principal concepts of the Bayesian decision procedure based on the modus-ponens rule are proposed. Some problems concerned with the practical realization of the fuzzy Bayesian method are considered '
	keywords = textrank.multiword_keywords( text )
	# keywords = textrank.rank( text )
	print( keywords )
	# plt.show()