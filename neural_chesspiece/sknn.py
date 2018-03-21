from sklearn.neural_network import MLPClassifier
from data_loader import load_dataset

path_train  = "/home/henry/workspace/SSE3/neural_chesspiece/data/"
dataset_train,dataset_test,i_size,o_size = load_dataset(path_train,1.0)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(400,), random_state=1)
clf.fit(dataset_train[0], dataset_train[1])    
# clf.predict([[1., 2.]])
# clf.predict([[0., 0.]])
from sklearn.externals import joblib
joblib.dump(clf, '/home/henry/workspace/SSE3/neural_chesspiece/skmodel.pkl') 
#clf = joblib.load('filename.pkl') 

