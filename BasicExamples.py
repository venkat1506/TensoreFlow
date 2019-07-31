import tensorflow as tf

a=tf.constant('hello world')
print(a)
b=tf.constant(100)
print(b)

sess=tf.Session()
print(sess.run(a))
print(sess.run(b))

c=tf.constant(4)
d=tf.constant(5)

add=c+d
sub=c-d
mul=c*d
sess=tf.Session()
print("constant values ")
print("addition:",sess.run(add))
print("subtraction:",sess.run(sub))
print("Multilation:",sess.run(mul))

new=tf.placeholder(tf.int64)
print(new)
new1=tf.placeholder(tf.int64)
print(new1)

add=tf.add(new,new1)
mul=tf.multiply(new,new1)

out={new:4,new1:5}

with tf.Session() as sess:
    print("place holders")
    print("addition",sess.run(add,feed_dict={new:4,new1:5}))
    print("subtract",sess.run(mul,feed_dict=out))


import numpy as np

a=np.array([[2.0,4.2]])
b=np.array([[4.2],[5.5]])

print(a.shape)
print(b.shape)

mat=tf.constant(a)
mat1=tf.constant(b)

matmul=tf.matmul(mat,mat1)

with tf.Session() as sess:
    result=sess.run(matmul)
    print("matrix multilation val:",result)
