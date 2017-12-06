import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)

# values are valuated in Sessions
sess = tf.Session()

# placeholders and lambda function definition
# the values of Variables are not initialized when declared, even if you specify the initial value
W = tf.Variable([.3],dtype= tf.float32)
b = tf.Variable([-.3],dtype= tf.float32)
x = tf.placeholder(tf.float32)

# multiply here has issues?
linear_model = W * x+b

# init here only a handle for the run() function to take to initialize
init = tf.global_variables_initializer()
sess.run(init)

#print(sess.run(linear_model,{x: [1,2,3,4]}))

y = tf.placeholder(tf.float32)

#define the loss function to be sum of square of error
loss = tf.reduce_sum(tf.square(linear_model - y))

print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# variable assignment action. create handle for the run function to use
#fixW = tf.assign(W,[-1.])
#fixb = tf.assign(b,[1.])
#sess.run([fixW,fixb])

# rerun the loass function result with new value of W and b
# at this point no optimization is done, we've just guessed the value for W and b
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# define optimizer, 0.01 is the step for all variables to change when adapting
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

# user needs to manually call trainer for multiple times for adaption
# variables must be changed on the road
# !!!: giving user ability to execute these single steps can enable them to do things like change the learning rate for each step
for i in range(5000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# print the variable final numbers
print(sess.run([W,b]))

