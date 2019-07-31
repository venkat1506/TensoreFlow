import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

shape=mnist.train.images.shape
newshape=mnist.train.images[4].reshape(28,28)
print(shape)
print(newshape)


import matplotlib.pyplot as plt
plt.imshow(newshape,cmap='Greys')
plt.show()

learning_rate=0.001
training_epochs=15
batch_size=100

n_classes=10
n_samples=mnist.train.num_examples
n_inputs=784
n_hidden_1=256
n_hidden_2=256


def multilayer_perceptron(x,weights,biases):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer=tf.matmul(layer_2,weights['out'])+biases['out']
    return out_layer

weights={
    'h1':tf.Variable(tf.random_normal([n_inputs,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder('float',[None ,n_inputs])
y = tf.placeholder('float',[None,n_classes])

predctyion=multilayer_perceptron(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predctyion,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#training model

t=mnist.train.next_batch(1)
xsamp,ysamp=t

plt.imshow(xsamp.reshape(28,28),cmap='Greys')
plt.show()


sess=tf.InteractiveSession()
init=tf.initialize_all_variables()
sess.run(init)

for epoch in range(training_epochs):

    avg_cost=0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        avg_cost +=c/total_batch
    print("Epoch:{} cost{:.4f}".format(epoch+1,avg_cost))
print("modal has completed {} epochs of training".format(training_epochs))


#model Evaluation

correct_prediction=tf.equal(tf.argmax(predctyion,1),tf.argmax(y,1))
print(correct_prediction)

correct_prediction=tf.cast(correct_prediction,'float')
print(correct_prediction)

accuricy=tf.reduce_mean(correct_prediction)
type(accuricy)

print(accuricy)


mnist.test.labels[0]
acc=accuricy.eval({x:mnist.test.images,y:mnist.test.labels})
print(acc)


