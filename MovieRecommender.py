from __future__ import print_function
"""
MovieRecommender
( My program of Exercise 8 of Andrew Ng's ML course in Python )
A recommender system via collaborative filtering.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from Imports import *
import ML

def ShowProgressDot ( x ) :
  print ( ".", end="" )
  sys.stdout.flush()
  return None

def main () :

  ## initialize random number generators and interactive plotting
  ML.Initialize()

  ## load data
  mat = scipy.io.loadmat ( "Data/ex8movies.mat", mat_dtype=True )
  Y = mat['Y'] # nMovies x nUsers matrix of [1,5]-star movie ratings
  R = mat['R'] # binary flag R[i,j] for whether movie i was rated by user j
  if Y.shape != R.shape : raise Exception()
  nMovies = Y.shape[0]
  nUsers  = Y.shape[1]
  print ( "%d movies" % nMovies )
  print ( "%d users"  % nUsers  )

  ## average rating for the first movie
  ratings = numpy.compress ( R[0,:], Y[0,:] )
  average = numpy.mean ( ratings )
  print ( "average rating for the first movie =", average )
  if abs(average - 3.878319) > 5e-7 : raise Exception()

  ## predict ratings for movies that users have not yet rated

  ## confirm the collaborative filtering cost function
  mat = scipy.io.loadmat ( "Data/ex8params.mat", mat_dtype=True )
  X = mat['X']
  Theta = mat['Theta']
  if     X.shape[0] !=     Y.shape[0] : raise Exception()
  if Theta.shape[0] !=     Y.shape[1] : raise Exception()
  if     X.shape[1] != Theta.shape[1] : raise Exception()
  nFeatures = 3
  weights = numpy.concatenate ( ( X[0:5,0:3].ravel(), Theta[0:4,0:3].ravel() ) )
  YY = Y[0:5,0:4]
  RR = R[0:5,0:4]

  ## no regularization
  Lambda = 0.0
  args = ( YY, RR, nFeatures, Lambda )
  J, G = ML.CollaborativeFilteringCostFunction ( weights, *args )
  print ( "test cost = %f (lambda=0.0)" % J )
  if abs(J - 22.2246037257) > 2e-11 : raise Exception()
  e = 1e-4
  numgrad = numpy.zeros ( weights.shape, dtype=float )
  perturb = numpy.zeros ( weights.shape, dtype=float )
  for p in range ( 0, weights.size ) :
    perturb[p] = e
    J1 = ML.CollaborativeFilteringCostFunction ( weights-perturb, *args ) [0]
    J2 = ML.CollaborativeFilteringCostFunction ( weights+perturb, *args ) [0]
    numgrad[p] = ( J2 - J1 ) / ( 2.0 * e )
    perturb[p] = 0.0
  diff = numpy.linalg.norm ( numgrad - G ) / numpy.linalg.norm ( numgrad + G )
  print ( "grad diff = %g" % diff )
  if abs(diff) >= 9.0e-13 : raise Exception()

  ## with regularization
  Lambda = 1.5
  args = ( YY, RR, nFeatures, Lambda )
  J, G = ML.CollaborativeFilteringCostFunction ( weights, *args )
  print ( "test cost = %f (lambda=1.5)" % J )
  if abs(J - 31.344056) > 3e-7 : raise Exception()
  e = 1e-4
  numgrad = numpy.zeros ( weights.shape, dtype=float )
  perturb = numpy.zeros ( weights.shape, dtype=float )
  for p in range ( 0, weights.size ) :
    perturb[p] = e
    J1 = ML.CollaborativeFilteringCostFunction ( weights-perturb, *args ) [0]
    J2 = ML.CollaborativeFilteringCostFunction ( weights+perturb, *args ) [0]
    numgrad[p] = ( J2 - J1 ) / ( 2.0 * e )
    perturb[p] = 0.0
  diff = numpy.linalg.norm ( numgrad - G ) / numpy.linalg.norm ( numgrad + G )
  print ( "grad diff = %g" % diff )
  if abs(diff) >= 1.65299e-12 : raise Exception()

  ## add a new user
  myRatings = numpy.zeros ( (nMovies), dtype=float )
  myRatings[  0] = 4
  myRatings[  6] = 3
  myRatings[ 11] = 5
  myRatings[ 53] = 4
  myRatings[ 63] = 5
  myRatings[ 65] = 3
  myRatings[ 68] = 5
  myRatings[ 97] = 2
  myRatings[182] = 4
  myRatings[225] = 5
  myRatings[354] = 5
  Y = numpy.c_[myRatings, Y]
  R = numpy.c_[(myRatings!=0.0), R]
  nMovies = Y.shape[0]
  nUsers  = Y.shape[1]

  ## normalize across users for each movie
  mu = numpy.zeros ( (nMovies), dtype=float )
  ## (there is a bug in the class code that doesn't pass Ynorm to fmincg)
  for movie in range ( 0, nMovies ) :
    idx = (R[movie,:] == 1.0).nonzero()
    mu[movie] = numpy.mean ( Y[movie,idx] )
    Y[movie,idx] -= mu[movie]

  ## train the collaborative filtering model
  ## (note that these random weights can differ from the class run)
  nFeatures = 10
  X = numpy.random.randn ( nMovies, nFeatures )
  Theta = numpy.random.randn ( nUsers, nFeatures )
  weights = numpy.concatenate ( ( X.ravel(), Theta.ravel() ) )
  Lambda = 10.0
  args = ( Y, R, nFeatures, Lambda )
 #ShowProgressDot = lambda x : print ( ".", end="", flush=True )
  result = scipy.optimize.minimize (
    ML.CollaborativeFilteringCostFunction, weights, args, method="CG", jac=True,
    options={'maxiter':100}, callback=ShowProgressDot
  )
  print("") # print a newline after progress dots
  #(scipy returns false if maxiter is reached)
  #if not result.success : raise Exception()

  ## X is a matrix of row vectors, nMovies-by-nFeatures
  weights = result.x
  b = 0
  e = b + nMovies * nFeatures
  r = numpy.arange ( b, e )
  X = numpy.reshape ( numpy.take(weights,r), (nMovies, nFeatures) )

  ## Theta is a matrix of row vectors, nUsers-by-nFeatures
  b = e
  e = b + nUsers * nFeatures
  Theta = numpy.reshape ( weights[b:e], (nUsers, nFeatures) )

  ## read movie names
  names = []
  text = open ( "Data/ex8movies.txt", "rt" )
  for line in text :
    names.append ( line.rstrip('\n') )
  text.close()

  ## predictions
  allPredictions = X.dot(Theta.T)
  myPredictions = allPredictions[:,0] + mu
  print ( "" )
  print ( "new top-rated movies ... " )
  idx = numpy.argsort ( -myPredictions ) # '-' makes ascending -> descending
  n = 0
  for i in range ( 0, nMovies ) :
    j = idx[i]
    s = numpy.sum ( R[j,:].ravel() )
    if s < 250 : continue # skip movies with too few ratings
    print ( "movie %4d:  %.0f (%g) %s" % ( j, myPredictions[j], s, names[j] ) )
    n += 1
    if n == 10 : break # how many top movies to show

  ## predictions for the originally rated movies change now,
  ## because we optimized the weights for all users instead of a specific user
  print ( "" )
  print ( "originally rated movies ... " )
  movies = numpy.array ( [0, 6, 11, 53, 63, 65, 68, 97, 182, 225, 354] )
  rating = numpy.array ( [4, 3,  5,  4,  5,  3,  5,  2,   4,   5,   5] )
  for i in range ( len(movies) ) :
    j = movies[i]
    print (
      "movie %3d:  %d -> %.0f %s" % ( j, rating[i], myPredictions[j], names[j] )
    )

################################################################################

if __name__ == "__main__" :
  main()


