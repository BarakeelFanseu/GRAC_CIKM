"""
JNMF:
Rundong Du, Barry L. Drake, and Haesun Park. 2017.
Hybrid Clustering based on Content and Connection Structure
using Joint Nonnegative Matrix Factorization.
CoRRabs/1703.09646 (2017). arXiv:1703.0964
"""


import numpy as np
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
from sys import exit
from NMF_Base import NMFBase
from metrics import clustering_metrics


class JNMF(NMFBase):
	"""
	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm
	"""
	
	def initialize_h_bar(self):
		""" Initalize W to random values [0,1]."""
		self.H_bar = np.random.random((self._rank, self._samples))
	
	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W') and hasattr(self, 'H_bar'):
			error = LA.norm(self.X - np.dot(self.W, self.H)) # + LA.norm(self.X - np.dot(self.H_bar.T, self.H))
		else:
			error = None

		return error

    
	def compute_graph(self, weight_type='heat-kernel', param=0.3):
		if weight_type == 'heat-kernel':
			samples = np.matrix(self.X.T)
			sigma= param
			A= np.zeros((samples.shape[0], samples.shape[0]))

			for i in range(A.shape[0]):
				for j in range(A.shape[1]):
					A[i][j]= np.exp(-(LA.norm(samples[i] - samples[j] ))/sigma )

			return A
		elif weight_type == 'dot-weighting':
			samples = np.matrix(self.X.T)
			A= np.zeros((samples.shape[0], samples.shape[0]))

			for i in range(A.shape[0]):
				for j in range(A.shape[1]):
					A[i][j]= np.dot(samples[i],samples[j])

			return A


	def compute_factors(self, max_iter=100, alpha=10, beta=10, weight_type='heat-kernel', adj_type=None, A=None, labels=None):
		
		if self.check_non_negativity():
			pass
		else:
			print("The given matrix contains negative values")
			exit()
       
		if not hasattr(self,'W'):
			self.initialize_w()

		if not hasattr(self,'H'):
			self.initialize_h()
		
		if not hasattr(self,'H_bar'):
			self.initialize_h_bar()
		
		# if  adj_type in ['clique', 'HyperAdj', 'precomputed']:
		# 	print('Using graph / Hypergraph Laplacian')
		# 	# D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))

		# elif adj_type=='HyperNcut':
		# 	print('JNMFL so I used instead of D')
		# 	# D = sp.eye(A.shape[0]).toarray()
			
		# else:
		# 	print('building manifold graph')
		# 	A = self.compute_graph(weight_type, param)
		# 	# D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))
			

		# self.frob_error = np.zeros(max_iter)

		for i in range(max_iter):
			print('iter: {}'.format(i))
				
			self.update_w(alpha, beta, A)
			self.update_h(alpha, beta, A)
			self.update_h_bar(alpha, beta, A)
			
			# self.frob_error[i] = self.frobenius_norm()
			
			predict_labels = np.asarray(np.argmax(self.H.T, axis=1)).squeeze()
			# print(predict_labels.shape)
			cm = clustering_metrics(labels, predict_labels)
			cm.evaluationClusterModelFromLabel()


	def update_h(self, alpha, beta, A):
		
		eps = 2**-8
		h_num = alpha*np.dot(self.H_bar, A) + beta*self.H_bar + np.dot(self.W.T, self.X)
		h_den = alpha*np.dot(np.dot(self.H_bar, self.H_bar.T), self.H) + beta*self.H + np.dot(np.dot(self.W.T, self.W), self.H)
		
		self.H = np.multiply(self.H, (h_num+eps)/(h_den+eps))
		self.H[self.H <= 0] = eps
		self.H[np.isnan(self.H)] = eps


	def update_w(self, alpha, beta, A):
		
		eps = 2**-8
		XH = self.X.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T)) + eps
		self.W *= XH
		self.W /= WHtH
		self.W[self.W <= 0] = eps
		self.W[np.isnan(self.W)] = eps


	def update_h_bar(self, alpha, beta, A):
		
		eps = 2**-8
		h_bar_num = alpha*self.H.dot(A) + beta*self.H
		h_bar_den = alpha*self.H.dot(self.H.T.dot(self.H_bar)) +  beta*self.H_bar + eps
		self.H = np.multiply(self.H_bar, (h_bar_num+eps)/(h_bar_den+eps))
		self.H_bar[self.H_bar <= 0] = eps
		self.H_bar[np.isnan(self.H_bar)] = eps