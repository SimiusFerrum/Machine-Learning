#!/usr/bin/env python2 -B -O
"""
LinearRegression.py
A linear-regression model to predict the president who will win california in 2016-17
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from Imports import *
import ML

def main () :

	## initialize random number generators and interactive plotting
	ML.Initialize()
	
	## extract training data from file 
	Xy = numpy.loadtxt ("PresDataCali.txt", dtype=float, delimiter=',' )
	
	## get # of training examples (m)
	m = Xy.shape[0]
	
	## get # of features (n)
	n = Xy.shape[1] - 1

	## get data from labels
	XDemo = Xy [ 0:6 :, n]
	print XDemo
	XRepu = Xy [ 6:m :, n]
	print XRepu
	yDemo = Xy [ 0:6 :, 0] 
	print yDemo
	yRepu = Xy [ 6:m :, 0]

	#plot
	matplotlib.pyplot.figure ( 1, facecolor="white" )
	matplotlib.pyplot.show   ( block=False )
	matplotlib.pyplot.title  ( "Scatter plot of training data" )
	matplotlib.pyplot.ylabel ( "Percentage of popular vote" )
	matplotlib.pyplot.xlabel ( "Year of election" )
	matplotlib.pyplot.axis   ( [1992,2014,0,100] )
	matplotlib.pyplot.xticks ( numpy.arange(1992,2014,4) )
	matplotlib.pyplot.plot   ( XDemo[:], yDemo[:], 'bx' )
	matplotlib.pyplot.plot   ( XRepu[:], yRepu[:], 'rx' )
	ML.Pause()


	## prepend ones to X's for bias
	XDemo = ML.PrependOnes ( XDemo )
	XRepu = ML.PrependOnes ( XRepu )
	n += 1

	## do mean normilization/feature scaling
	XD, muD, sgD = ML.Normalize ( XDemo )
	XR, muR, sgR = ML.Normalize ( XRepu )

	#-----------------------------------------------------------------------

	## initialize optimization data
	Lambda = 0.0
	argsD = ( XD, yDemo, Lambda )
	argsR = ( XR, yRepu, Lambda )
	thetaD = numpy.zeros ( (n), dtype=float )
	thetaR = numpy.zeros ( (n), dtype=float )

	## calculate the initial cost
	JinitialD = ML.LinearRegressionCostFunction ( thetaD, *argsD ) [0]
	JinitialR = ML.LinearRegressionCostFunction ( thetaR, *argsR ) [0]
	print JinitialD
	print JinitialR

	## apply gradient descent
	alpha = 0.01 
	num_iterations = 1500
	JD = numpy.empty ( ( num_iterations, 1 ) )
	JR = numpy.empty ( ( num_iterations, 1 ) )
	for iter in range ( 0, num_iterations ) :
		JD[iter], GD = ML.LinearRegressionCostFunction ( thetaD, *argsD )
		thetaD = thetaD - alpha * GD
		
	for iter in range ( 0, num_iterations ) :
		JR[iter], GR = ML.LinearRegressionCostFunction ( thetaR, *argsR )
		thetaR = thetaR - alpha * GR

	## calculate the final cost
	JDfinal = ML.LinearRegressionCostFunction ( thetaD, *argsD ) [0]
	JRfinal = ML.LinearRegressionCostFunction ( thetaR, *argsR ) [0]
	
	## print final theta value
	print ( "thetaD=", thetaD )
	print ( "thetaR=", thetaR )

	## try some predictions 
	prediction1D = 10000.0 * numpy.inner ( [1.0, (3.5-muR[1])/sgR[1]], thetaD )
	prediction1R = 10000.0 * numpy.inner ( [1.0, (3.5-muR[1])/sgR[1]], thetaR )
	##print ( "Prediction1D=", prediction1D )
	##print ( "Prediction1R=", prediction1R )
	if abs ( prediction1D + 13987001.865835916 ) >= 0.5 : raise Exception()
	if abs ( prediction1R - 16321182.870398071 ) >= 0.5 : raise Exception()
	prediction2D = 10000.0 * numpy.inner ( [1.0, (7.0-muD[1])/sgD[1]], thetaD )
	prediction2R = 10000.0 * numpy.inner ( [1.0, (7.0-muR[1])/sgR[1]], thetaR )
	##print ( "Prediction2D=", prediction2D )
	##print ( "Prediction2R=", prediction2R )
	if abs ( prediction2D + 13961554.373054251 ) >= 0.5 : raise Exception()
	if abs ( prediction2R - 16293370.37828725 ) >= 0.5 : raise Exception()

	## calculate hypothesis 
	hD = XD.dot(thetaD)
	hR = XR.dot(thetaR)

	## plot the training data with the linear regression line
	matplotlib.pyplot.figure ( 2, facecolor="white" )
	matplotlib.pyplot.show   ( block=False )
	matplotlib.pyplot.title  ( "California populous' votes" )
	matplotlib.pyplot.ylabel ( "Percent of popular vote" )
	matplotlib.pyplot.xlabel ( "Year of election" )
	matplotlib.pyplot.axis   ( [1992,2014,0,100] )
	matplotlib.pyplot.xticks ( numpy.arange(1992,2014,4) )
	matplotlib.pyplot.plot   ( sgD[1]*XD[:,1]+muD[1], yDemo[:], 'bx', label="Democratic populous vote percentage" )
	matplotlib.pyplot.plot   ( sgD[1]*XD[:,1]+muD[1], hD[:], 'b-', label="Linear regression for Demo's" )
	matplotlib.pyplot.plot   ( sgR[1]*XR[:,1]+muR[1], yRepu[:], 'rx', label="Republican populous vote percentage" )
	matplotlib.pyplot.plot   ( sgR[1]*XR[:,1]+muR[1], hR[:], 'r-', label="Linear regression for Repub's" )
	matplotlib.pyplot.legend ( loc="lower right", numpoints=1, fontsize=11, borderaxespad=2 )
	matplotlib.pyplot.draw   ()
	ML.Pause()		
################################################################################
if __name__ == "__main__" :
	main()
