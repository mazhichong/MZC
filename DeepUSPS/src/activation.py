import tensorflow as tf
def taylor_softmax(inputs, order=6):          #order——taylor order
    exponent_tensor = taylor_exp(inputs, order)
    sum_of_exponents = tf.reduce_sum(exponent_tensor,axis=-1,keepdims=True)
    result = tf.divide(exponent_tensor, sum_of_exponents)
    return result

def taylor_exp(x, order):
  x_shape= tf.shape(x)
  temp   = tf.ones(x_shape)
  result = tf.ones(x_shape)
  denom  = tf.ones(x_shape)
  for i in range(1, int(order + 1)):
    temp   = tf.multiply(temp, x)
    iteri  = tf.fill(x_shape, i)
    iteri  = tf.cast(iteri, tf.float32)
    denom  = tf.multiply(denom, iteri)
    result = result + (temp / denom)
  return result


