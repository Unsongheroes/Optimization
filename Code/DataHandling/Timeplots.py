"""
time fit pca mnist:  1.9403796195983887
time fit pca orl:  0.016350269317626953
time fit pca 95 mnist:  7.1845057010650635
time fit pca 95 orl:  0.22893166542053223
time transform mnist train pca:  0.3002936840057373
time transform mnist test pca:  0.050862789154052734
time transform orl train pca:  0.001993894577026367
time transform orl test pca:  0.0019948482513427734
time transform mnist train pca 95:  0.5460562705993652
time transform orl train pca 95:  0.0029892921447753906
time transform mnist test pca 95:  0.08776640892028809
time transform orl test pca 95:  0.0029926300048828125
NearestCentroid fit mnist time:  0.0937492847442627
NearestCentroid acc mnist time:  0.05684709548950195
NCCC - MNIST Full Accuracy:  0.8203
NearestCentroid fit mnist pca time:  0.011969566345214844
NearestCentroid acc mnist pca time:  0.0019943714141845703
NCCC - MNIST PCA Accuracy:  0.4365
NearestCentroid fit mnist pca 95% time:  0.08477425575256348
NearestCentroid acc mnist pca 95% time:  0.008064508438110352
NCCC - MNIST PCA 95% Accuracy:  0.8199
NearestCentroid fit orl time:  0.0040323734283447266
NearestCentroid acc orl time:  0.0009961128234863281
NCCC - ORL full Accuracy:  0.9166666666666666
NearestCentroid fit orl pca time:  0.0010323524475097656
NearestCentroid acc orl pca time:  0.0009975433349609375
NCCC - ORL PCA Accuracy:  0.4083333333333333
NearestCentroid fit orl pca 95 time:  0.0009968280792236328
NearestCentroid acc orl pca 95 time:  0.0009958744049072266
NCCC - MNIST PCA 95% Accuracy:  0.9166666666666666 """


""" KNN - MNIST fit time:  32.479533672332764
KNN - MNIST acc time:  930.0407781600952
KNN - MNIST Full Accuracy:  0.9691
KNN - MNIST PCA fit time:  0.13463687896728516
KNN - MNIST PCA acc time:  0.36539745330810547
KNN - MNIST PCA Accuracy:  0.3899
KNN - MNIST PCA 95 fit time:  5.156649589538574
KNN - MNIST PCA 95 acc time:  143.09227991104126
KNN - MNIST PCA 95 Accuracy:  0.9694
KNN - ORL fit time:  0.06083798408508301
KNN - ORL acc time:  0.11722970008850098
KNN - ORL Full Accuracy:  0.9916666666666667
KNN - ORL PCA fit time:  0.06083798408508301
KNN - ORL PCA acc time:  0.0071337223052978516
KNN - ORL PCA Accuracy:  0.3
KNN - ORL PCA 95 fit time:  0.06083798408508301
KNN - ORL PCA 95 acc time:  0.0071337223052978516
KNN - ORL PCA 95 Accuracy:  0.9916666666666667 """

import numpy as np
import matplotlib.pyplot as plt
class TimeDataVisualiser:
    def barvisualizer(self, trainingtime, perdiction, name):
        ind = np.arange(3)    # the x locations for the groups
        width = 0.35
        p1 = plt.bar(ind, trainingtime, width)
        p2 = plt.bar(ind, perdiction, width, bottom=trainingtime)

        plt.ylabel("time in seconds")
        plt.xticks(ind, ("Full", "PCA", "PCA95"))
        plt.legend((p1[0], p2[1]), ("Training time", "prediction and scoring time"))
        plt.savefig(name,dpi=100)

    def barvisualizeracc(self, full, PCA, PCA95, name):
        ind = np.arange(3)
        width = 0.35
        plt.bar(ind, (full, PCA,PCA95),width)
        plt.ylabel("acc in procent(%)")
        plt.xticks(ind,("Full", "PCA", "PCA95"))
        plt.savefig(name,dpi=100)

    def barvisualizerpreprocess(self, pcafit, pcatrans, pca95fit, pca95trans, name):
        ind = np.arange(2)
        width = 0.35
        p1 = plt.bar(ind, (pcafit, pca95fit), width)
        p2 = plt.bar(ind, (pcatrans, pca95trans), width, bottom=(pcafit, pca95fit))

        plt.ylabel("Time in seconds")
        plt.xticks(ind, ("PCA", "PCA95"))
        plt.legend((p1[0], p2[1]), ("Fitting the model", "Transforming the data"))
        plt.savefig(name,dpi=100)
