__author__ = "Copyright (C) 2016 by Jared Levy"

from Imports import *

################################################################################

def InitializeRNGs ( seed ) :
  """ Initialize random number generators.
  """
  random.seed ( seed )
  numpy.random.seed ( seed )
  return None

################################################################################

def Initialize ( seed=12345 ) :
  """ Initialize various packages for use with this module.
  """
  # initialize random number generators
  InitializeRNGs ( seed )
  # turn-on interactive plotting
  matplotlib.pyplot.ion()
  return None

################################################################################

class ClassificationTreeTemp :
  def __init__ ( self ) :
    self.lefCnd = []
    self.rigCnd = []
    self.lefLbl = []
    self.rigLbl = []
    self.lefSmp = []
    self.rigSmp = []

class ClassificationTreeNode :
  def __init__ ( self ) :
    self.rss = float('inf')
    self.idx = -1
    self.val = float('nan')
    self.lefMaj = float('nan')
    self.rigMaj = float('nan')
    self.lefNode = None
    self.rigNode = None
  def predict ( self, x ) :
    if x[self.idx] == self.val :
      if self.lefNode :
        return self.lefNode.predict ( x )
      else :
        return self.lefMaj
    else :
      if self.rigNode :
        return self.rigNode.predict ( x )
      else :
        return self.rigMaj

class ClassificationTree :
  def __init__ ( self ) :
    self.root = ClassificationTreeNode()
    self.oob  = []
  def predict ( self, x ) :
    return self.root.predict ( x )

def BuildClassificationTreeNode ( node, X, y, samples, nFeatures ) :

  m = len(samples)
  n = X.shape[1]
  Xs = X[samples,:]
  ys = y[samples]
  F = numpy.random.choice ( n, size=nFeatures, replace=False )
  temp = ClassificationTreeTemp()

  for f in F :

    Xf = Xs[:,f]
    #S = numpy.unique ( Xf )
    S = set ( Xf[:] )

    for s in S :

      lefCnd = ( Xf == s )
      rigCnd = ( Xf != s )
      lefLbl = ys[lefCnd]
      rigLbl = ys[rigCnd]

      lefCnt = collections.defaultdict(int)
      rigCnt = collections.defaultdict(int)
      for lbl in lefLbl :
        lefCnt[lbl] += 1
      for lbl in rigLbl :
        rigCnt[lbl] += 1

      lefTot = float(len(lefLbl))
      rigTot = float(len(rigLbl))

      lefRss = 0.0
      rigRss = 0.0
      lefFrc = 0
      rigFrc = 0
      lefMaj = None
      rigMaj = None

      if 0 :
        ## Gini index/impurity (better for noisy high-dimensional data)
        for lbl, cnt in lefCnt.items() :
          frc = float(cnt) / lefTot
          lefRss += ( frc * frc )
          if frc > lefFrc :
            lefFrc = frc
            lefMaj = lbl
        lefRss = 1.0 - lefRss
        for lbl, cnt in rigCnt.items() :
          frc = float(cnt) / rigTot
          rigRss += ( frc * frc )
          if frc > rigFrc :
            rigFrc = frc
            rigMaj = lbl
        rigRss = 1.0 - rigRss
      else :
        ## Cross-entropy (better for clean low-dimensional data)
        ## ("information gain" is entropyBeforeSplit - entropyAfterSplit)
        for lbl, cnt in lefCnt.items() :
          frc = float(cnt) / lefTot
          lefRss -= frc * math.log ( frc, 2.0 )
          if frc > lefFrc :
            lefFrc = frc
            lefMaj = lbl
        for lbl, cnt in rigCnt.items() :
          frc = float(cnt) / rigTot
          rigRss -= frc * math.log ( frc, 2.0 )
          if frc > rigFrc :
            rigFrc = frc
            rigMaj = lbl
      ## There is also variance reduction (better for continuous variables)

      tot = lefTot + rigTot
      rss = lefRss * (lefTot/tot) + rigRss * (rigTot/tot)

      if ( rss < node.rss ) :
        node.rss = rss
        node.idx = f
        node.val = s
        node.lefMaj = lefMaj
        node.rigMaj = rigMaj
        temp.lefCnd = lefCnd
        temp.rigCnd = rigCnd
        temp.lefLbl = lefLbl
        temp.rigLbl = rigLbl

  lefNum = len(temp.lefLbl)
  if lefNum > 0 :
    temp.lefSmp = samples[temp.lefCnd]

  rigNum = len(temp.rigLbl)
  if rigNum > 0 :
    temp.rigSmp = samples[temp.rigCnd]

  if lefNum > 1 :
    if not numpy.all ( X[temp.lefSmp,:] == X[temp.lefSmp[0],:] ) :
      if not numpy.all ( y[temp.lefSmp] == y[temp.lefSmp[0]] ) :
        node.lefNode = ClassificationTreeNode()
        BuildClassificationTreeNode (
          node.lefNode, X, y, temp.lefSmp, nFeatures
        )

  if rigNum > 1 :
    if not numpy.all ( X[temp.rigSmp,:] == X[temp.rigSmp[0],:] ) :
      if not numpy.all ( y[temp.rigSmp] == y[temp.rigSmp[0]] ) :
        node.rigNode = ClassificationTreeNode()
        BuildClassificationTreeNode (
          node.rigNode, X, y, temp.rigSmp, nFeatures
        )

  return None

def BuildClassificationTree ( X, y, samples, nFeatures ) :
  tree = ClassificationTree()
  BuildClassificationTreeNode ( tree.root, X, y, samples, nFeatures )
  m = X.shape[0]
  tree.oob = numpy.ones ( m, dtype=bool )
  U = numpy.unique ( samples )
  for u in U :
    tree.oob[u] = False
  return tree

################################################################################

