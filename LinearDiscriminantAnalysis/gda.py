import numpy as np


class GaussianDiscriminantAnalysis:

    def __init__(self):
        
        self.numClasses = None
        self.prioriProbabilities = []
        self.means = []
        self.covs = []
    

    
    def fit(self,X,y):

        
        self.ClassNames = np.unique(y)
        self.numClasses = len(self.ClassNames)

        for i , className   in enumerate(self.ClassNames):
            CurrentClass = X[y == self.ClassNames[i]]
            self.prioriProbabilities.append( CurrentClass.shape[0] / X.shape[0])

            self.means.append( np.mean( CurrentClass , axis= 0))

            self.covs.append( np.cov(CurrentClass , rowvar=False))
        
        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.prioriProbabilities = np.array(self.prioriProbabilities)




    def predict_logproba(self,X):
        '''
        Predicts the log probability of X belonging to each class
        '''

        logproba = np.zeros(self.numClasses)
        for j in range(self.numClasses):
                logproba[j] = np.log(self.prioriProbabilities[j]) + np.log( multivariate_normal_gaussian(X,self.means[j],self.covs[j]) )
                
        
        return logproba
        
    def predict(self,X_test):
        X_test = np.array(X_test)
        pred_all = []
        for i in range(X_test.shape[0]):
    
            prediction = self.predict_logproba(X_test[i,:])

            pred_all.append( self.ClassNames[np.argmax(prediction)])
        
        return np.array(pred_all)
    
    def evaluate(self,X_test , y_test):

        pred = self.predict(X_test)

        y_test = np.array(y_test)

        pred = np.reshape(pred,(X_test.shape[0],1))
        y_test = np.reshape(y_test,(X_test.shape[0],1))
        
        acc = np.sum( pred == y_test )/ y_test.shape[0]

        print("Accuracy on this dataset is {} %".format(round(acc,4)*100))
        return pred , acc



def multivariate_normal_gaussian(X, mu, sigma):

    if sigma.size != 1 :
        part1 = np.power(2*np.pi,mu.shape[0]/2) * np.sqrt( np.linalg.det(sigma) )
        
        part2 = -0.5 * (X-mu).T @ np.linalg.inv(sigma) @ (X-mu)
        
        
    else:
        part1 = np.sqrt(sigma * 2 * np.pi)

        part2 = -0.5 * np.square(X - mu) / sigma

    prob = np.exp(part2) / part1
    return prob