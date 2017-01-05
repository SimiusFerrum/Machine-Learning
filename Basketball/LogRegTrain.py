"""
LogRegTrain.py
A logistic-regression classifier to predict the chance I make a shot from varying distances.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from   Imports import *
import LogisticRegressionLib 

def main () :
  ## uses a logistical regression library I wrote
  ## initialize random number generators and interactive plotting
  LogisticRegressionLib.Initialize()

  ## load the training examples
  Xy = numpy.loadtxt ( "Data/Shots.txt", dtype=float, delimiter=',' )

  ## get the number of training examples
  m = Xy.shape[0]

  ## get the number of features
  n = Xy.shape[1] - 1

  ## separate the data from the labels
  ## X = matrix of training examples with m rows and n features
  ##     rows=: (all)    cols=[0,n) (0 through n-1)
  ## y = vector of training example labels with m rows
  ##     rows=: (all)    cols=[n]   (the n'th column)
  ## notice that X is a matrix and y is a vector
  X = Xy [ :, 0:n ]
  y = Xy [ :,   n ]
  assert X.shape == (m,n)
  assert y.shape == (m,)

  ## plot the data
  ## extract the rows of X where y==1
  Xpos = X[numpy.where(y.ravel()==1),:][0]
  ypos = y[numpy.where(y.ravel()==1)]
  ## extract the rows of X where y==0
  Xneg = X[numpy.where(y.ravel()==0),:][0]
  yneg = y[numpy.where(y.ravel()==0)]
  matplotlib.pyplot.figure ( 1, facecolor="white" )
  matplotlib.pyplot.show   ( block=False )
  matplotlib.pyplot.title  ( "Scatter plot of training data" )
  matplotlib.pyplot.xlabel ( "Distance from hoop" )
  matplotlib.pyplot.ylabel ( "Made / Missed" )
  matplotlib.pyplot.axis   ( [-5, 35, -0.5, 1.5] )
  matplotlib.pyplot.plot   ( Xpos[:], ypos[:], 'k+', label="Made",
    linewidth=2, markersize=7
  )
  matplotlib.pyplot.plot   ( Xneg[:], yneg[:], 'ko', label="Missed",
    markerfacecolor='y', markersize=7
  )
  matplotlib.pyplot.legend ( loc="upper right", numpoints=1, fontsize=11 )
  matplotlib.pyplot.draw   ()
  LogisticRegressionLib.Pause()

  ## save the un-prepended/normalized version for plotting later
  Xu = X

  ## prepend a column of ones to X for bias
  X = LogisticRegressionLib.PrependOnes ( X )
  n += 1

  ## perform mean normalization and feature scaling
  ## mu = average of each column of X
  ## sg = sigma (standard deviation) of each column of X
  X, mu, sg = LogisticRegressionLib.Normalize ( X )
  ## print ( "mu =", mu )
  ## print ( "sg =", sg )

  #-----------------------------------------------------------------------------

  ## initialize the optimization data
  Lambda = 0.0
  args = ( X, y, Lambda )
  theta = numpy.zeros ( (n), dtype=float )

  ## optimize theta
  ## if the CG method doesn't converge, try BFGS instead
  result = scipy.optimize.minimize (
    LogisticRegressionLib.LogisticRegressionCostFunction, theta, args, method="BFGS", jac=True,
  )
  if not result.success : raise Exception()
  print ( "number of iterations = %d" % result.nit )
  theta = result.x
  ## print ( "theta =", theta )

  ## try a 'made' prediction at distance=10
  x = numpy.array ( [1, (10-mu[1])/sg[1]], dtype=float )
  prediction = LogisticRegressionLib.Sigmoid(x.dot(theta))
  print ( "prediction at 10 feet = %.3f" % prediction )

  ## try a 'missed' prediction at distance=20
  x = numpy.array ( [1, (20-mu[1])/sg[1]], dtype=float )
  prediction = LogisticRegressionLib.Sigmoid(x.dot(theta))
  print ( "prediction at 20 feet = %.3f" % prediction )

  ## plot the sigmoid (the logistic function)
  h = LogisticRegressionLib.Sigmoid(X.dot(theta))
  ## print ( "Xu =", Xu )
  ## print ( "X  =", X  )
  ## print ( "h  =", h )
  matplotlib.pyplot.figure ( 2, facecolor="white" )
  matplotlib.pyplot.show   ( block=False )
  matplotlib.pyplot.title  ( "Sigmoid Function" )
  matplotlib.pyplot.xlabel ( "Distance from hoop" )
  matplotlib.pyplot.ylabel ( "Made / Missed Probability" )
  matplotlib.pyplot.axis   ( [-5, 35, -0.5, 1.5] )
  matplotlib.pyplot.plot   ( Xu[:,0], h[:], 'b-', label="Model", linewidth=2 )
  matplotlib.pyplot.plot   ( Xpos[:], ypos[:], 'k+', label="Made",
    linewidth=2, markersize=7
  )
  matplotlib.pyplot.plot   ( Xneg[:], yneg[:], 'ko', label="Missed",
    markerfacecolor='y', markersize=7
  )
  matplotlib.pyplot.legend ( loc="upper right", numpoints=1, fontsize=11 )
  matplotlib.pyplot.draw   ()
  
  ## ask for user input and make a prediction with the input
  userInp = input ( "Enter a distance from the hoop " )
  inputMath = numpy.array ( [1, (userInp-mu[1])/sg[1]], dtype=float )
  inputProbability = LogisticRegressionLib.Sigmoid(inputMath.dot(theta))
  print ( "The probability is: %.3f" % inputProbability )
  
  LogisticRegressionLib.Pause()

################################################################################

if __name__ == "__main__" :
  main()

