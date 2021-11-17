# Please cite our paper if you use our code.

## Below are the ways to run the different methods for Attributed Hypergraph clustering (AHC) used in the paper:
    (1) for all models, see parameters in paper.

    (2) To run GRAC use:
        python GRAC.py --kwargs

    (3) To run JNMF use:
        python running_JNMF.py --kwargs

    (4) To run GNMF-L, GNMF-clique and GNMF-A use:
        python running_GNMF --kwargs

    (5) To run Kmeans and other methods which donot combine graph structure and node attributes use:
        python others_clean.py --kwargs

# Please refference the papers below if you use any of the models here

[1] Rundong Du, Barry L. Drake, and Haesun Park. 2017. Hybrid Clustering based on Content and Connection Structure using Joint Nonnegative Matrix Factorization. CoRRabs/1703.09646 (2017). arXiv:1703.0964

[2] Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562). Paper

[3] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative Matrix Factorization, Nature 401(6755), 788-799. Paper

[4] Cai, D., He, X., Han, J., & Huang, T. S. (2011). Graph regularized nonnegative matrix factorization for data representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(8), 1548-1560. Paper

[5] Bayar, B., Bouaynaya, N., & Shterenberg, R. (2014). Probabilistic non-negative matrix factorization: theory and application to microarray data analysis. Journal of bioinformatics and computational biology, 12(01), 1450001. Paper

[6] Zhang, D., Zhou, Z. H., & Chen, S. (2006). Non-negative matrix factorization on kernels. PRICAI 2006: Trends in Artificial Intelligence, 404-412. Paper

[7] Yanez, Felipe, and Francis Bach. "Primal-dual algorithms for non-negative matrix factorization with the Kullback-Leibler divergence." In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on, pp. 2257-2261. IEEE, 2017. Paper

# Acknowledgements
[1] Satwik Bhattamishra @https://github.com/satwik77 for the Open Source libnmf library

[2] Yadati Naganand for providing the HyperGCN code @https://github.com/malllabiisc/HyperGCN


# Please Cite Our Work:
@inproceedings{barak2021hypergraph,
  title={HyperGraph Convolution Based Attributed HyperGraph Clustering},
  author={Barakeel Fanseu Kamhoua and Lin Zhang and Kaili Ma and James Cheng and Bo Li and Bo Han},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={},
  year={2021}
}

