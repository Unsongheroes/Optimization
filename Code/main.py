

# imports
# importing data
from DataHandling.DataLoading import Datasets
#Used for visualising data
from DataHandling.Dataplotting import DataVisualiser
#for plotting
from matplotlib import pyplot as plt
#nearest subclass classifier 
from DataHandling.Algorithm import NearstSubClassCentroidClassifer
#ploting bar plots
from DataHandling.Timeplots import TimeDataVisualiser
# to split the ORL data set
from sklearn.model_selection import train_test_split
# to reduce the dataset into 2d. 
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestCentroid
#timing of the models training and scoring
import time
#Used for getting the current directory (Main)
import pathlib
#Used as the dataobject for data
import numpy as np 

#init class
datasets = Datasets()
Datavisualiser = DataVisualiser()
NSCC = NearstSubClassCentroidClassifer()
timeplot = TimeDataVisualiser()


#Base path for imports
base_path = str(pathlib.Path(__file__).parent.absolute()) + '/'

#Load MNIST dataset
mnist_path = "../Data/MNIST/"
mnist_img_train, mnist_lbl_train, mnist_img_test, mnist_lbl_test = Datasets.load_mnist(datasets, base_path, mnist_path)

#Load ORL dataset
orl_path = "../Data/ORL_txt/"
orl_img, orl_lbl = Datasets.load_orl(datasets,base_path, orl_path)

#split dataset
orl_img_train, orl_img_test, orl_lbl_train, orl_lbl_test = train_test_split(orl_img, orl_lbl, test_size=0.3)

#transform data using PCA
#transform MNIST data
pca = PCA(n_components=2)
pca_orl = PCA(n_components=2)
""" PCA_none = PCA() """
PCA_orl_none = PCA()
pca_95 = PCA(n_components=0.95)
pca_orl_95 = PCA(n_components=0.95)

#reshape the dimentionality for the dataset
mnist_img_train_flatten_2d = mnist_img_train.reshape((mnist_img_train.shape[0], mnist_img_train.shape[1]*mnist_img_train.shape[2]))
mnist_img_test_flatten_2d = mnist_img_test.reshape((mnist_img_test.shape[0], mnist_img_test.shape[1]*mnist_img_test.shape[2]))
orl_img_train_flatten_2d = orl_img_train.reshape((orl_img_train.shape[0], orl_img_train.shape[1]*orl_img_train.shape[2]))
orl_img_test_flatten_2d = orl_img_test.reshape((orl_img_test.shape[0], orl_img_test.shape[1]*orl_img_test.shape[2]))

#fitting the model on the training set the pca object holds this model.
timepcafitmniststart = time.time()
pca.fit(mnist_img_train_flatten_2d)
timepcafitmniststop = time.time()
print('time fit pca mnist: ',timepcafitmniststop - timepcafitmniststart)
timepcafitorlstart = time.time()
pca_orl.fit(orl_img_train_flatten_2d)
timepcafitorlstop = time.time()
print('time fit pca orl: ',timepcafitorlstop - timepcafitorlstart)
timepcafitmnist95start = time.time()
pca_95.fit(mnist_img_train_flatten_2d)
timepcafitmnist95stop = time.time()
print('time fit pca 95 mnist: ',timepcafitmnist95stop - timepcafitmnist95start)
timepcafitorl95start = time.time()
pca_orl_95.fit(orl_img_train_flatten_2d)
timepcafitorl95stop = time.time()
print('time fit pca 95 orl: ',timepcafitorl95stop - timepcafitorl95start)

# use for modelling 
""" PCA_none.fit(mnist_img_train_flatten_2d) """
""" PCA_orl_none.fit(orl_img_train_flatten_2d) """
# preforms the dimentionality for the flattened datasets.

timestartmnisttrainPCA = time.time()
mnist_img_train_PCA = pca.transform(mnist_img_train_flatten_2d)
timestopmnisttrainPCA = time.time()
print('time transform mnist train pca: ',timestopmnisttrainPCA-timestartmnisttrainPCA)
timestartmnisttestPCA = time.time()
mnist_img_test_PCA = pca.transform(mnist_img_test_flatten_2d)
timestopmnisttestPCA = time.time()
print('time transform mnist test pca: ',timestopmnisttestPCA-timestartmnisttestPCA)
timestartORLtrainPCA = time.time()
orl_img_train_PCA = pca_orl.transform(orl_img_train_flatten_2d)
timestopORLtrainPCA = time.time()
print('time transform orl train pca: ', timestopORLtrainPCA-timestartORLtrainPCA)
timestartOrltestPCA = time.time()
orl_img_test_PCA = pca_orl.transform(orl_img_test_flatten_2d)
timestopOrltestPCA = time.time()
print('time transform orl test pca: ', timestopOrltestPCA-timestartOrltestPCA)
timestartmnisttrainPCA95 = time.time()
mnist_img_train_PCA_95 = pca_95.transform(mnist_img_train_flatten_2d)
timestopmnisttrainPCA95 = time.time()
print('time transform mnist train pca 95: ', timestopmnisttrainPCA95-timestartmnisttrainPCA95)
timestartorltrainPCA95 = time.time()
orl_img_train_PCA_95 = pca_orl_95.transform(orl_img_train_flatten_2d)
timestoporltrainpca95 = time.time()
print('time transform orl train pca 95: ', timestoporltrainpca95-timestartorltrainPCA95)
timestartmnisttestPCA95 = time.time()
mnist_img_test_PCA_95 = pca_95.transform(mnist_img_test_flatten_2d)
timestopmnisttestpca95 = time.time()
print('time transform mnist test pca 95: ',timestopmnisttestpca95-timestartmnisttestPCA95)
timestartorltestPCA95 = time.time()
orl_img_test_PCA_95 = pca_orl_95.transform(orl_img_test_flatten_2d)
timestoporltestPCA95 = time.time()
print('time transform orl test pca 95: ',timestoporltestPCA95-timestartorltestPCA95)
#timing preprocessing
""" pcafit = timepcafitmniststop - timepcafitmniststart
pcatrans = timestopmnisttrainPCA-timestartmnisttrainPCA+timestopmnisttestPCA-timestartmnisttestPCA
pca95fit = timepcafitmnist95stop - timepcafitmnist95start
pca95trans = timestopmnisttrainPCA95-timestartmnisttrainPCA95 + timestopmnisttestpca95-timestartmnisttestPCA95
timeplot.barvisualizerpreprocess(pcafit,pcatrans,pca95fit, pca95trans, "preprocessing_MNIST.png") """
""" pcafit = timepcafitorlstop - timepcafitorlstart
pcatrans = timestopORLtrainPCA-timestartORLtrainPCA+timestopOrltestPCA-timestartOrltestPCA
pca95fit = timepcafitorl95stop - timepcafitorl95start
pca95trans = timestoporltrainpca95-timestartorltrainPCA95 + timestoporltestPCA95-timestartorltestPCA95
timeplot.barvisualizerpreprocess(pcafit,pcatrans,pca95fit, pca95trans, "preprocessing_ORL.png") """
#---------------------- Visualize Data ----------------------
#------ Images ------
#MNIST Train
""" for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(mnist_img_train[i])
#plt.savefig('MNIST_Data_sample.png', dpi=100)
plt.show()
for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(orl_img_train[i])
#plt.savefig('ORL_Data_sample.png', dpi=100)
plt.show() """
""" DataVisualiser.plotImages(Datavisualiser, mnist_img_train[0:11], mnist_lbl_train[0:11])
#MNIST Test
DataVisualiser.plotImages(Datavisualiser, mnist_img_test[0:9], mnist_lbl_test[0:9])

#ORL
DataVisualiser.plotImages(Datavisualiser, orl_img_train[0:9], orl_lbl_train[0:9]) """



#------ 2D Plot -----
#MNIST Train
sample = mnist_img_train_PCA[:10]
samplelbl = mnist_lbl_train[:10]
DataVisualiser.plot2dData(Datavisualiser, sample, labels=samplelbl)
""" #MNIST Test
DataVisualiser.plot2dData(Datavisualiser, mnist_img_test_PCA, labels=mnist_lbl_test) """
#Orl
""" DataVisualiser.plot2dData(Datavisualiser, orl_img_train_PCA, labels=orl_lbl_train) """

# PCA optimization

""" plt.plot(np.cumsum(PCA_none.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('cumsum_PCA.png', dpi=100) """
""" plt.plot(np.cumsum(PCA_orl_none.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('cumsum_orl_PCA.png', dpi=100) """

""" print(mnist_img_train_PCA_95.shape)
print(orl_img_train_PCA_95.shape) """

#------------------------ Exercises ------------------------
#Exercise 1 - Nearest Class centroid classifier
#--------MNIST--------
#Full dimensionality
clf = NearestCentroid()
""" timestartclffitmnisttrain = time.time()
clf.fit(mnist_img_train_flatten_2d, mnist_lbl_train)
timestopclffitmnisttrain = time.time()
print("NearestCentroid fit mnist time: ", timestopclffitmnisttrain-timestartclffitmnisttrain)
timestartclfaccmnisttrain = time.time()
accurary = clf.score(mnist_img_test_flatten_2d,mnist_lbl_test)
timestopclfaccmnisttrain = time.time()
print("NearestCentroid acc mnist time: ",timestopclfaccmnisttrain-timestartclfaccmnisttrain)
print("NCCC - MNIST Full Accuracy: ",accurary)
#PCA applied
timestartclffitmnistPCA = time.time()
clf.fit(mnist_img_train_PCA, mnist_lbl_train)
timestopclffitmnistPCA = time.time()
print("NearestCentroid fit mnist pca time: ",timestopclffitmnistPCA-timestartclffitmnistPCA)
timestartclfaccmnistPCA = time.time()
PCA_accuray = clf.score(mnist_img_test_PCA, mnist_lbl_test)
timestopclfaccmnistPCA = time.time()
print("NearestCentroid acc mnist pca time: ",timestopclfaccmnistPCA-timestartclfaccmnistPCA)
print("NCCC - MNIST PCA Accuracy: ", PCA_accuray)
#PCA_95 applied
timestartclffitmnistPCA95 = time.time()
clf.fit(mnist_img_train_PCA_95, mnist_lbl_train)
timestopclffitmnistPCA95 = time.time()
print("NearestCentroid fit mnist pca 95% time: ",timestopclffitmnistPCA95 -timestartclffitmnistPCA95)
timestartclfaccmnistPCA95 = time.time()
PCA_95_accuray = clf.score(mnist_img_test_PCA_95, mnist_lbl_test)
timestopclfaccmnistPCA95 = time.time()
print("NearestCentroid acc mnist pca 95% time: ",timestopclfaccmnistPCA95 - timestartclfaccmnistPCA95)
print("NCCC - MNIST PCA 95% Accuracy: ", PCA_95_accuray) """
# plotting times for MNIST NCC
""" trainingtimes = (timestopclffitmnisttrain-timestartclffitmnisttrain, timestopclffitmnistPCA-timestartclffitmnistPCA, timestopclffitmnistPCA95 -timestartclffitmnistPCA95)
predictions = (timestopclfaccmnisttrain-timestartclfaccmnisttrain, timestopclfaccmnistPCA-timestartclfaccmnistPCA,timestopclfaccmnistPCA95 - timestartclfaccmnistPCA95)
timeplot.barvisualizer(trainingtimes,predictions, "timing_MNIST_NCC.png") """
#plotting acc for MNIST NCC
""" timeplot.barvisualizeracc(accurary*100, PCA_accuray*100, PCA_95_accuray*100, "acc_NCC_MNIST.png") """
# ORL
""" timestartclffitorl = time.time()
clf.fit(orl_img_train_flatten_2d, orl_lbl_train)
timestopclffitorl = time.time()
print("NearestCentroid fit orl time: ", timestopclffitorl-timestartclffitorl)
timestartclfaccorl = time.time()
orl_accurary = clf.score(orl_img_test_flatten_2d,orl_lbl_test)
timestopclfaccorl = time.time()
print("NearestCentroid acc orl time: ", timestopclfaccorl-timestartclfaccorl)
print("NCCC - ORL full Accuracy: ", orl_accurary)
#PCA applied
timestartclffitorlpca = time.time()
clf.fit(orl_img_train_PCA, orl_lbl_train)
timestopclffitorloca = time.time()
print("NearestCentroid fit orl pca time: ", timestopclffitorloca-timestartclffitorlpca)
timestartclfaccorlpca = time.time()
orl_PCA_accurary = clf.score(orl_img_test_PCA,orl_lbl_test)
timestopclfaccorlpca = time.time()
print("NearestCentroid acc orl pca time: ", timestopclfaccorlpca-timestartclfaccorlpca)
print("NCCC - ORL PCA Accuracy: ", orl_PCA_accurary)
#PCA_95 applied
timestartclffitorlpca95 = time.time()
clf.fit(orl_img_train_PCA_95, orl_lbl_train)
timestopclffitorloca95 = time.time()
print("NearestCentroid fit orl pca 95 time: ", timestopclffitorloca95-timestartclffitorlpca95)
timestartclfaccorlpca95 = time.time()
PCA_orl_95_accuray = clf.score(orl_img_test_PCA_95, orl_lbl_test)
timestopclfaccorlpca95 = time.time()
print("NearestCentroid acc orl pca 95 time: ", timestopclfaccorlpca95-timestartclfaccorlpca95)
print("NCCC - ORL PCA 95% Accuracy: ", PCA_orl_95_accuray) """
# plotting times for ORL NCC
""" trainingtimes = (timestopclffitorl-timestartclffitorl, timestopclffitorloca-timestartclffitorlpca, timestopclffitorloca95-timestartclffitorlpca95)
predictions = (timestopclfaccorl-timestartclfaccorl, timestopclfaccorlpca-timestartclfaccorlpca,timestopclfaccorlpca95-timestartclfaccorlpca95)
timeplot.barvisualizer(trainingtimes,predictions, "timing_ORL_NCC.png")  """
#plotting acc for ORL NCC
""" timeplot.barvisualizeracc(orl_accurary*100, orl_PCA_accurary*100, PCA_orl_95_accuray*100, "acc_NCC_ORL.png") """


#Exercise 2 - Nearest Sub Class centroid classifier
#--------MNIST--------
sets = [5]
for nr_sub_classes in sets:
    """ timestartNSCCfitmnist = time.time()
    model = NSCC.fit(mnist_img_train_flatten_2d, mnist_lbl_train, nr_sub_classes)
    timestopNSCCfitmnist = time.time()
    print("NSCCC(",nr_sub_classes,") time fit mnist: ", timestopNSCCfitmnist-timestartNSCCfitmnist)
    timestartNSCCaccmnist = time.time()
    accuracy_MNIST = NSCC.score(model, mnist_img_test_flatten_2d, mnist_lbl_test)
    timestopNSCCaccmnist = time.time()
    print("NSCCC(",nr_sub_classes,") time acc mnist: ", timestopNSCCaccmnist-timestartNSCCaccmnist)
    print("NSCCC(",nr_sub_classes,")  - MNIST Full Accuracy: ", accuracy_MNIST)
    timestartNSCCfitmnistPCA = time.time()
    model_PCA = NSCC.fit(mnist_img_train_PCA, mnist_lbl_train, nr_sub_classes)
    timestopNSCCfitmnistPCA = time.time()
    print("NSCCC(",nr_sub_classes,") time fit mnist pca: ", timestopNSCCfitmnistPCA-timestartNSCCfitmnistPCA)
    timestartNSCCaccmnistPCA = time.time()
    accuracy_PCA_MNIST = NSCC.score(model_PCA, mnist_img_test_PCA, mnist_lbl_test)
    timestopNSCCaccmnistPCA = time.time()
    print("NSCCC(",nr_sub_classes,") time acc mnist pca: ", timestopNSCCaccmnistPCA-timestartNSCCaccmnistPCA)
    print("NSCCC(",nr_sub_classes,")  - MNIST PCA Accuracy: ", accuracy_PCA_MNIST)
    timestartNSCCfitmnistPCA95 = time.time()
    model_PCA_95 = NSCC.fit(mnist_img_train_PCA_95, mnist_lbl_train, nr_sub_classes)
    timestopNSCCfitmnistPCA95 = time.time()
    print("NSCCC(",nr_sub_classes,") time fit mnist pca 95: ", timestopNSCCfitmnistPCA95-timestartNSCCfitmnistPCA95)
    timestartNSCCaccmnistpca95 = time.time()
    accuracy_PCA_95_MNIST = NSCC.score(model_PCA_95, mnist_img_test_PCA_95, mnist_lbl_test)
    timestopNSCCaccmnistpca95 = time.time()
    print("NSCCC(",nr_sub_classes,") time acc mnist pca 95: ", timestopNSCCaccmnistpca95-timestartNSCCaccmnistpca95)
    print("NSCCC(",nr_sub_classes,")  - MNIST PCA 95 Accuracy: ", accuracy_PCA_95_MNIST) """
    # plotting time
    """ trainingtimes = (timestopNSCCfitmnist-timestartNSCCfitmnist, timestopNSCCfitmnistPCA-timestartNSCCfitmnistPCA, timestopNSCCfitmnistPCA95-timestartNSCCfitmnistPCA95)
    predictions = (timestopNSCCaccmnist-timestartNSCCaccmnist, timestopNSCCaccmnistPCA-timestartNSCCaccmnistPCA,timestopNSCCaccmnistpca95-timestartNSCCaccmnistpca95)
    timeplot.barvisualizer(trainingtimes,predictions, "timing_MNIST_NSCC_5.png")  """
    #plotting acc for ORL NCC
    """ timeplot.barvisualizeracc(accuracy_MNIST*100, accuracy_PCA_MNIST*100, accuracy_PCA_95_MNIST*100, "acc_NSCC_MNIST_5.png") """
# ORL
    """ timestartNSCCfitorl = time.time()
    model_ORL = NSCC.fit(orl_img_train_flatten_2d, orl_lbl_train,nr_sub_classes)
    timestopNSCCfitorl = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl fit: ", timestopNSCCfitorl-timestartNSCCfitorl)
    timestartNSCCaccorl = time.time()
    accurary_ORL = NSCC.score(model_ORL,orl_img_test_flatten_2d, orl_lbl_test)
    timestopNSCCaccorl = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl acc: ", timestopNSCCaccorl-timestartNSCCaccorl)
    print("NSCCC(",nr_sub_classes,")  - ORL Full Accuracy: ", accurary_ORL)
    timestartNsCCfitorlPCA = time.time()
    model_ORL_PCA = NSCC.fit(orl_img_train_PCA, orl_lbl_train, nr_sub_classes)
    timestopNSCCfitorlPCA = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl pca fit: ", timestopNSCCfitorlPCA-timestartNsCCfitorlPCA)
    timestartNSCCaccorlPCA = time.time()
    accuracy_ORL_PCA = NSCC.score(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test)
    timestopNSCCaccorlPCA = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl pca acc: ", timestopNSCCaccorlPCA-timestartNSCCaccorlPCA)
    print("NSCCC(",nr_sub_classes,")  - ORL PCA Accuracy: ", accuracy_ORL_PCA)
    timestartfitNSCCorlPCA95 = time.time()
    model_ORL_PCA_95 = NSCC.fit(orl_img_train_PCA_95, orl_lbl_train, nr_sub_classes)
    timestopfitNSCCorlPCA95 = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl pca 95 fit: ", timestopfitNSCCorlPCA95-timestartfitNSCCorlPCA95)
    timestartNSCCaccorlPCA95 = time.time()
    accuracy_ORL_PCA_95 = NSCC.score(model_ORL_PCA_95, orl_img_test_PCA_95, orl_lbl_test)
    timestopNSCCaccorlPCA95 = time.time()
    print("NSCCC(",nr_sub_classes,")  - time orl pca 95 acc: ", timestopNSCCaccorlPCA95-timestartNSCCaccorlPCA95)
    print("NSCCC(",nr_sub_classes,")  - ORL PCA 95 Accuracy: ", accuracy_ORL_PCA_95) """
    # plotting time
    """ trainingtimes = (timestopNSCCfitorl-timestartNSCCfitorl, timestopNSCCfitorlPCA-timestartNsCCfitorlPCA, timestopfitNSCCorlPCA95-timestartfitNSCCorlPCA95)
    predictions = (timestopNSCCaccorl-timestartNSCCaccorl, timestopNSCCaccorlPCA-timestartNSCCaccorlPCA,timestopNSCCaccorlPCA95-timestartNSCCaccorlPCA95)
    timeplot.barvisualizer(trainingtimes,predictions, "timing_ORL_NSCC_5.png") """
    #plotting acc for ORL NCC
    """ timeplot.barvisualizeracc(accurary_ORL*100, accuracy_ORL_PCA*100, accuracy_ORL_PCA_95*100, "acc_NSCC_ORL_5.png") """

#Exercise 3 - Nearest Neighbour classifier
KNN = KNeighborsClassifier(n_neighbors=1)
timestartKNNmnist = time.time()
KNN.fit(mnist_img_train_flatten_2d, mnist_lbl_train)
timestopKNNmnist = time.time()
print("KNN - MNIST fit time: ", timestopKNNmnist-timestartKNNmnist)
timestartKNNaccmnist = time.time()
accurary_MNIST_KNN = KNN.score(mnist_img_test_flatten_2d, mnist_lbl_test)
timestopKNNaccmnist = time.time()
print("KNN - MNIST acc time: ", timestopKNNaccmnist-timestartKNNaccmnist)
print("KNN - MNIST Full Accuracy: ", accurary_MNIST_KNN)
timestartKNNfitMnistPCA = time.time()
KNN.fit(mnist_img_train_PCA, mnist_lbl_train)
timestopKNNfitMnistPCA = time.time()
print("KNN - MNIST PCA fit time: ", timestopKNNfitMnistPCA-timestartKNNfitMnistPCA)
timestartKNNaccMnistPCA = time.time()
accurary_PCA_KNN = KNN.score(mnist_img_test_PCA, mnist_lbl_test)
timestopKNNaccMNistPCA = time.time()
print("KNN - MNIST PCA acc time: ", timestopKNNaccMNistPCA-timestartKNNaccMnistPCA)
print("KNN - MNIST PCA Accuracy: ", accurary_PCA_KNN)
timestartKNNfitMNistPCA95 = time.time()
KNN.fit(mnist_img_train_PCA_95, mnist_lbl_train)
timestopKNNfitMnistPCA95 = time.time()
print("KNN - MNIST PCA 95 fit time: ", timestopKNNfitMnistPCA95-timestartKNNfitMNistPCA95)
timestartKNNaccMnistPCA95 = time.time()
accurary_PCA_95_KNN = KNN.score(mnist_img_test_PCA_95, mnist_lbl_test)
timestopKNNaccMnistPCA95 = time.time()
print("KNN - MNIST PCA 95 acc time: ", timestopKNNaccMnistPCA95-timestartKNNaccMnistPCA95)
print("KNN - MNIST PCA 95 Accuracy: ", accurary_PCA_95_KNN)
# time plotting
""" trainingtimes = (timestopKNNmnist-timestartKNNmnist, timestopKNNfitMnistPCA-timestartKNNfitMnistPCA, timestopKNNfitMnistPCA95-timestartKNNfitMNistPCA95)
predictions = (timestopKNNaccmnist-timestartKNNaccmnist, timestopKNNaccMNistPCA-timestartKNNaccMnistPCA,timestopKNNaccMnistPCA95-timestartKNNaccMnistPCA95)
timeplot.barvisualizer(trainingtimes,predictions, "timing_MNIST_KNN.png") """
#plotting acc for MNIST KNN
""" timeplot.barvisualizeracc(accurary_MNIST_KNN*100, accurary_PCA_KNN*100, accurary_PCA_95_KNN*100, "acc_KNN_MNIST.png") """
#ORL
""" timestartKNNfitORL = time.time()
KNN.fit(orl_img_train_flatten_2d, orl_lbl_train)
timestopKNNfitORL = time.time()
print("KNN - ORL fit time: ", timestopKNNfitORL-timestartKNNfitORL)
timestartKNNaccORL = time.time()
accurary_ORL_KNN = KNN.score(orl_img_test_flatten_2d, orl_lbl_test)
timestopKNNaccOrl = time.time()
print("KNN - ORL acc time: ", timestopKNNaccOrl-timestartKNNaccORL)
print("KNN - ORL Full Accuracy: ", accurary_ORL_KNN)
timestartKNNfitORLPCA = time.time()
KNN.fit(orl_img_train_PCA, orl_lbl_train)
timestopKNNfitORLPCA = time.time()
print("KNN - ORL PCA fit time: ", timestopKNNfitORLPCA-timestartKNNfitORLPCA)
timestartKNNaccORLPCA = time.time()
accurary_ORL_PCA_KNN = KNN.score(orl_img_test_PCA, orl_lbl_test)
timestopKNNaccORLPCA = time.time()
print("KNN - ORL PCA acc time: ", timestopKNNaccORLPCA-timestartKNNaccORLPCA)
print("KNN - ORL PCA Accuracy: ", accurary_ORL_PCA_KNN)
timestartKNNfitorlPCA95 = time.time()
KNN.fit(orl_img_train_PCA_95, orl_lbl_train)
timestopKNNfitorlPCA95 = time.time()
print("KNN - ORL PCA 95 fit time: ", timestopKNNfitorlPCA95-timestartKNNfitorlPCA95)
timestartKNNaccorlPCA95 = time.time()
accurary_ORL_PCA_95_KNN = KNN.score(orl_img_test_PCA_95, orl_lbl_test)
timestopKNNaccorlPCA95 = time.time()
print("KNN - ORL PCA 95 acc time: ", timestopKNNaccorlPCA95-timestartKNNaccorlPCA95)
print("KNN - ORL PCA 95 Accuracy: ", accurary_ORL_PCA_95_KNN) """
# time plotting
""" trainingtimes = (timestopKNNfitORL-timestartKNNfitORL, timestopKNNfitORLPCA-timestartKNNfitORLPCA, timestopKNNfitorlPCA95-timestartKNNfitorlPCA95)
predictions = (timestopKNNaccOrl-timestartKNNaccORL, timestopKNNaccORLPCA-timestartKNNaccORLPCA,timestopKNNaccorlPCA95-timestartKNNaccorlPCA95)
timeplot.barvisualizer(trainingtimes,predictions, "timing_ORL_KNN.png") """
#plotting acc for ORL NCC
""" timeplot.barvisualizeracc(accurary_ORL_KNN*100, accurary_ORL_PCA_KNN*100, accurary_ORL_PCA_95_KNN*100, "acc_KNN_ORL.png") """

