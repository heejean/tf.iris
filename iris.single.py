#!/usr/bin/python

import tensorflow as tf
import pandas     as pd
import numpy      as np

#########################
#                       #
#  Data Initialization  #
#                       #
#########################

num_of_step = 200
dfrm = pd.read_csv('iris.csv')

#
# Convert Data.Frame to numpy array
#

data = dfrm.to_csv( header = False, index = False ).split('\n')

feature = np.asarray( list ( map ( lambda x : [ float(y) for y in x.split(',')[:-1] ], data[:-1] ) ) )
class_item = [ y for y in set( map ( lambda x : x.split(',')[-1], data[:-1] ) ) ]

num_of_class    = len( class_item )
num_of_data     = len( feature )
num_of_feature  = len( feature[0] )

class_map     = { class_item[i] : i for i in range( num_of_class ) }
class_inv_map = { i : class_item[i] for i in range( num_of_class ) }


# Get result with Softmax type

cls_tmp = np.asarray( list ( map ( lambda x : class_map.get( x.split(',')[-1] ), data[:-1] ) ) )
result  = np.asarray( list ( map ( lambda x : [ int( x == i ) for i in range( num_of_class ) ], cls_tmp ) ) )


#########################
#                       #
#    TensorFlow Init    #
#                       #
#########################

#
# Declaration
#

x = tf.placeholder( tf.float32, [ None, num_of_feature ] )
W = tf.Variable( tf.zeros( [ num_of_feature, num_of_class ] ) )
b = tf.Variable( tf.zeros( [ num_of_class ] ) )

y  = tf.nn.softmax( tf.matmul( x, W ) + b )
y_ = tf.placeholder( tf.float32, [ None, num_of_class ] )

cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_ * tf.log(y) , reduction_indices = [1] ) )
train_step = tf.train.GradientDescentOptimizer( 0.1 ).minimize( cross_entropy )


#
# Initialization
#

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run( init )


#########################
#                       #
#    Training           #
#                       #
#########################

for i in range( num_of_step ):
    sess.run( train_step, feed_dict = { x: feature, y_: result } )

    correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ))

    print( '\r', i, ':', sess.run( accuracy, feed_dict = { x: feature, y_: result } ), end = '' )
    
print()


#########################
#                       #
#     Get Result        #
#                       #
#########################

W_s = sess.run(W)
b_s = sess.run(b)


#########################
#                       #
#     Verification      #
#                       #
#########################

Weight   = np.matrix( W_s )
Bias     = np.matrix( b_s )
Feature  = np.matrix( feature )
Result   = ( Feature * Weight + Bias ).tolist()
Class    = list( map ( lambda x : class_inv_map.get(np.argmax(x)), Result ) )
Original = list( map ( lambda x : x.split(',')[-1], data[:-1] ) )

print( [ [ i, orig, clas ] for i, orig, clas in zip( range( min( len(Original), len(Class) ) ), Original, Class ) if orig != clas ] )


#print( '\n'.join( [ str(elem) for elem in [ [ i, orig, clas ] for i, orig, clas in zip( range( min( len(Original), len(Class) ) ), Original, Class ) ] ] ) )

#print ( min( len(Original), len(Class) ) )
#print ( [ range( min( len(Original), len(Class) ) )] )

#print('Weight :\n', Weight )
#print('Bias :\n', Bias )
#print('Feature :\n', Feature )
#print('Result :\n', Result, type(Result) )
#print('Class :\n', Class, type(Class) )
#print('Original:\n', Original, type(Original) )
