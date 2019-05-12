#   ***********************************************************************
#   Test of the Matrix multiplication Implementation with
#   gradient checking
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************

import tensorflow as tf
import twodlearn.core.ops as tdl

with tf.Session('') as session:
    n = 10
    m = 20
    l = 5

    a_mat = tf.Variable(tf.truncated_normal([n, m], stddev=0.1), name='a_mat')
    b_mat = tf.Variable(tf.truncated_normal([m, l], stddev=0.1), name='b_mat')

    my_product = tdl.my_matmul(a_mat, b_mat)
    product = tf.matmul(a_mat, b_mat)

    diff = tf.nn.l2_loss(product - my_product)

    tf.initialize_all_variables().run()

    print("\n\n --------------- running my_matmul_test2 -----------------")
    [d_val, my_c, c] = session.run([diff, my_product, product])

    print("\n\n --------------- output -----------------")
    print("error:", d_val)
    print("output shape:", c.shape)

    print("\n\n --------------- gradient check -----------------")
    loss = tf.reduce_sum(my_product)
    #gradient_error = tf.test.compute_gradient_error( [a_mat, b_mat], [a_mat.get_shape(), b_mat.get_shape()] ,loss, [1])
    gradient_error = tf.test.compute_gradient_error(
        [a_mat, b_mat], [(n, m), (m, l)], loss, [1])

    print("Gradient error: ", gradient_error)
    print("\n")
