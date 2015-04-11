#!/usr/bin/env python

from similaritymeasures import Similarity

def main():

	""" main function to create Similarity class instance and get use of it """
	
	measures = Similarity()

	print measures.euclidean_distance([0,3,4,5],[7,6,3,-1])
	print measures.jaccard_similarity([0,1,2,5,6],[0,2,3,5,7,9])

if __name__ == "__main__":
	main()