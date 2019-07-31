import tensorflow as tf
x1 = tf.add(4,8,)
x2 = tf.multiply(x1,5)
x3 = tf.add(12,6,)
x4 = tf.multiply(x3,x2)
x5 = tf.div(x4,2)

with tf.Session() as sess:
  output = sess.run(x5)
  print(output)

with  tf.Session() as sess:
  outnew=tf.summary.FileWriter("./logs/add",sess.graph)    #tensorboard --logdir=logs
  print(sess.run(x5))
  outnew.close()
