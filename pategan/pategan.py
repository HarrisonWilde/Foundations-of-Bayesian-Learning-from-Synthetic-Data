'''PATE-GAN function'''

# Necessary packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression


def pate_lamda(x, teacher_models, lamda):
  '''Returns PATE_lambda(x).
  
  Args:
    - x: feature vector
    - teacher_models: a list of teacher models
    - lamda: parameter
    
  Returns:
    - n0, n1: the number of label 0 and 1, respectively
    - out: label after adding laplace noise.
  '''
      
  y_hat = list()
        
  for teacher in teacher_models:            
    temp_y = teacher.predict(np.reshape(x, [1,-1]))
    y_hat = y_hat + [temp_y]
  
  y_hat = np.asarray(y_hat)
  n0 = sum(y_hat == 0)
  n1 = sum(y_hat == 1)
  
  lap_noise = np.random.laplace(loc=0.0, scale=lamda)
  
  out = (n1+lap_noise) / float(n0+n1)
  out = int(out>0.5)
        
  return n0, n1, out 


def pategan(X_train, y_train, X_test, y_test, epsilon, delta, n_student_iter, num_teachers, batch_size, no_split, lamda, seed):
  '''Basic PATE-GAN framework.
  
  Args:
    - X_train, y_train: training data
    - X_test, y_test: testing data
    - parameters: PATE-GAN parameters
      - n_student_iter: the number of student training iterations
      - batch_size: the number of batch size for training student and generator
      - num_teachers: the number of teachers
      - epsilon, delta: Differential privacy parameters
      - lamda: noise size
      
  Returns:
    - X_train_hat, y_train: generated training data by differentially private generator
    - X_test_hat, y_test: generated training data by differentially private generator
  '''
  
  np.random.seed(seed)
  tf.random.set_random_seed(seed)

  # Reset the graph
  tf.reset_default_graph()

  
  # Other parameters
  # alpha initialize
  L = 20
  alpha = np.zeros([L])
  # initialize epsilon_hat
  epsilon_hat = 0
    
  # Network parameters
  no, dim = X_train.shape
  if not no_split:
    no_test, _ = X_test.shape
  # Random sample dimensions
  z_dim = int(dim)
  # Student hidden dimension
  student_h_dim = int(dim)
  # Generator hidden dimension
  generator_h_dim = int(4 * dim)  
  
  ## Partitioning the data into num_teachers subsets
  X_partition = list()
  partition_data_no = int(no / num_teachers)
  if not no_split:
    X_partition_test = list()
    partition_data_no_test = int(no_test / num_teachers)
  
  idx = np.random.permutation(no)
  if not no_split:
    idx_test = np.random.permutation(no_test)
    
  for i in range(num_teachers):
    temp_idx = idx[int(i*partition_data_no):int((i+1)*partition_data_no)]
    temp_X = X_train[temp_idx, :]      
    X_partition = X_partition + [temp_X]
    
    if not no_split:
      temp_idx_test = idx_test[int(i * partition_data_no_test):int((i+1)*partition_data_no_test)]
      temp_X_test = X_test[temp_idx_test, :]
      X_partition_test = X_partition_test + [temp_X_test]
  
  ## Necessary Functions for buidling NN models
  # Xavier Initialization Definition
  def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)    
        
  # Sample from uniform distribution
  def sample_Z(m, n):
    return np.random.uniform(0., 1., size = [m, n])
     
  ## Placeholder
  # PATE labels
  Y = tf.placeholder(tf.float32, shape = [None, 1])  
  # Random Variable    
  Z = tf.placeholder(tf.float32, shape = [None, z_dim])
   
  ## NN variables   
  # Student
  S_W1 = tf.Variable(xavier_init([dim, student_h_dim]))
  S_b1 = tf.Variable(tf.zeros(shape=[student_h_dim]))
    
  S_W2 = tf.Variable(xavier_init([student_h_dim,1]))
  S_b2 = tf.Variable(tf.zeros(shape=[1]))

  theta_S = [S_W1, S_W2, S_b1, S_b2]
    
  # Generator

  G_W1 = tf.Variable(xavier_init([z_dim, generator_h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

  G_W2 = tf.Variable(xavier_init([generator_h_dim,generator_h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

  G_W3 = tf.Variable(xavier_init([generator_h_dim,dim]))
  G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

  ## Models
  def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        
    return G_out
    
  def student(x):
    S_h1 = tf.nn.relu(tf.matmul(x, S_W1) + S_b1)
    S_out = tf.matmul(S_h1, S_W2) + S_b2
        
    return S_out
      
  ## Loss  
  G_sample = generator(Z)
  S_fake = student(G_sample)
  
  S_loss = tf.reduce_mean(Y * S_fake) - tf.reduce_mean((1-Y) * S_fake)
  G_loss = -tf.reduce_mean(S_fake)
  
  # Optimizer
  S_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
              .minimize(-S_loss, var_list=theta_S))
  G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
              .minimize(G_loss, var_list=theta_G))
  
  clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_S]
  
  ## Sessions
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
        
  ## Iterations
  while epsilon_hat < epsilon:      
    
    print(epsilon_hat, epsilon)
    # 1. Train teacher models
    teacher_models = list()
    
    for _ in range(num_teachers):
                
      Z_mb = sample_Z(partition_data_no, z_dim)
      G_mb = sess.run(G_sample, feed_dict = {Z: Z_mb})
                
      temp_X = X_partition[i]
      idx = np.random.permutation(len(temp_X[:, 0]))
      X_mb = temp_X[idx[:partition_data_no], :]
                
      X_comb = np.concatenate((X_mb, G_mb), axis = 0)
      Y_comb = np.concatenate((np.ones([partition_data_no,]), 
                               np.zeros([partition_data_no,])), axis = 0)
                
      model = LogisticRegression()
      model.fit(X_comb, Y_comb)
      teacher_models = teacher_models + [model]
            
    # 2. Student training
    for _ in range(n_student_iter):
          
      Z_mb = sample_Z(batch_size, z_dim)
      G_mb = sess.run(G_sample, feed_dict = {Z: Z_mb})
      Y_mb = list()
            
      for j in range(batch_size):                
        n0, n1, r_j = pate_lamda(G_mb[j, :], teacher_models, lamda)
        Y_mb = Y_mb + [r_j]
       
        # Update moments accountant
        q = np.log(2 + lamda * abs(n0 - n1)) - np.log(4.0) - \
            (lamda * abs(n0 - n1))
        q = np.exp(q)
                
        # Compute alpha
        for l in range(L):
          temp1 = 2 * (lamda**2) * (l+1) * (l+2)
          temp2 = (1-q) * ( ((1-q)/(1-q*np.exp(2*lamda)))**(l+1) ) + \
                  q * np.exp(2*lamda * (l+1))
          alpha[l] = alpha[l] + np.min([temp1, np.log(temp2)])
        
      # PATE labels for G_mb  
      Y_mb = np.reshape(np.asarray(Y_mb), [-1,1])
                
      # Update student
      _, D_loss_curr, _ = sess.run([S_solver, S_loss, clip_S], feed_dict = {Z: Z_mb, Y: Y_mb})
    
    # Generator Update        
    Z_mb = sample_Z(batch_size, z_dim)
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: Z_mb})
    print(np.mean(Y_mb))
        
    # epsilon_hat computation
    curr_list = list()        
    for l in range(L):
      temp_alpha = (alpha[l] + np.log(1/delta)) / float(l+1)
      curr_list = curr_list + [temp_alpha]
        
    epsilon_hat = np.min(curr_list)
    print(epsilon_hat, epsilon)        

  ## Outputs
  X_train_hat = sess.run([G_sample], feed_dict = {Z: sample_Z(no, z_dim)})[0]
  
  if no_split:

    return X_train_hat, y_train

  else:
  
    X_test_hat = sess.run([G_sample], feed_dict = {Z: sample_Z(no_test, z_dim)})[0]
    return X_train_hat, y_train, X_test_hat, y_test
