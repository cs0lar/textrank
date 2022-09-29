from itertools import combinations
import math 

import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

import networkx as nx

class TextRank( object ):
	"""
		It implements the Textrank algorithm from:

			“TextRank: Bringing Order into Texts"
			R. Mihalcea, P. Tarau, 2004

		Parameters:
		-----------
		
			N - the size of the window, in lexical units (e.g. words, sentences) that defines 
				the co-occurrence relation. Default is 2, i.e. a window that comprises the target 
				token as well as the two logical units either side of it.

			pos - an array of syntactic filters, codes for the parts of speech that are to be considered 
				  nodes of the graph (e.g. only use nouns and verbs). Default is noun and adjectives.
			
			T - the number of top ranked lexical units to return. It T is not specified, T defaults to a third of the vertices in the graph

	"""
	def __init__( self, N=2, pos=[ 'NN', 'JJ' ] ):
		
		self._N = N 
		self._pos = pos

	def preprocess( self, text ):
		"""
		It tokenises the input text and annotates it with parts of speech tags. The tags are used
		for filtering the tokens and retaining those lexical units (e.g. noun, adjectives) that
		will constitute the vertices of the text graph.

		Parameters:
		------------

		text: string
			the original text to analyse


		Returns
		--------

		tokens: list
			the list of tokens extracted from the original text

		vertices: list
			the subset of tokens that constitute the vertices for the text graph sorted in alphabetical order
		
		"""
		
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
		"""
		It performs keywords extraction using the TextRank algorithm on the input text.

		Parameters:
		------------
		
		text: string
			the original text to analyse

		tol: float
			the tolerance to use for convergence of the pagerank algorithm that scores the graph's vertices


		Returns
		--------
		ranking: list
			an ordered (descending) list of pairs, each of which is comprised of a keyword (token) and its pagerank score

		"""
		tokens, units = self.preprocess( text )

		self._lenunits = len( units )
		
		nxgraph = self.graph( units, tokens )

		scores = nx.pagerank( nxgraph, tol=tol )

		unsorted_ranking = [ ( units[ vertex ], score ) for vertex, score in scores.items() ]
		
		return sorted( unsorted_ranking, key=lambda x: x[ 1 ], reverse=True )


	def graph( self, vertices, tokens, plot=False ):
		"""
		It generates the text graph given a set of tokens and the subset of tokens
		that ought to be considered vertices of the graph.
		Given two vertices X and Y, an edge is added between them if and only if X and Y
		co-occur within a window of N lexical units in the tokenised text.
		The graph returned is un-directed (the in-degree of a vertex is equal to the out-degree) and
		un-weighted.
		The resulting graph can be plotted for visual inspection by setting `plot` to True. The plot
		function uses Matplotlib internally therefore in order to show the visualisation the 
		`matplotlib.pyplot` module must be loaded the `show()` function invoked.

		Parameters
		-----------

		vertices: list
			subset of tokens to be considered the vertices of the graph 

		tokens: list
			the list of tokens extracted from the original text

		plot: boolean
			if set to True, the generated graph is plotted using Matplotlib


		Returns
		-------

		graph: networkx graph
			an undirected, unweighted graph representing the lexical units co-occurrences in the original text

		"""
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
		"""
		It perform keywords extraction from the input text using the TextRank algorithm.

		Parameters
		-----------

		text: string
			the original text to analyse

		T: the number of top ranked keywords to return

		Returns
		-------

		keywords: list
			A number T of keywords extracted from the original text sorted in descending order according to their 
			TextRank score. If T was not specified, T is set to one third of the number of vertices in the text graph is returned

		"""
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
		"""
		It performs multi-word keyword extraction from the input text using the TextRank algorithm. A keyword 
		extraction is first applied and sequences of keywords that are found to be adjacent in the original 
		text are merged together. Un-merged keywords are retained as is.
		Multi-word keywords are scored by summing the scores of their component keywords.

		Parameters
		----------
		
		text: string
			the original text to be analysed
	
		T: the number of top ranked multi-word keywords to return

		Returns
		--------

		multikeywords: list
			A number T of multi-word keywords extracted from the original text sorted in descending order according
			to their TextRank score. If T was not specified, T is set to one third of the number of vertices in the text graph is returned


		"""
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

