import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

def np_exp_q(x,q=1):
    if q==1:
        return np.exp(x)
    else:
        exp_q_x = np.full_like(x,np.inf)
        exp_q_x[1+(1-q)*x > 0] = np.power(1+(1-q)*x[1+(1-q)*x > 0],1/(1-q))
        return exp_q_x

def np_log_q(x,q=1):
    if q==1:
        log_q_x = np.full_like(x,-np.inf)
        log_q_x[x>0] = np.log(x[x>0])
        return log_q_x
    else:
        log_q_x = np.full_like(x,-np.inf)
        log_q_x[x>0] = (np.power(x[x>0],1-q)-1)/(1-q)
        return log_q_x
    
def np_max_single_q(q_logit, q=1.):
    q_logit = np.reshape(q_logit,[-1,])
    max_q_logit = np.max(q_logit)
    safe_q_logit = q_logit - max_q_logit
    if q==1.:
        maxq = np.log(np.sum(np.exp(safe_q_logit))) + max_q_logit
        pq = np.exp(safe_q_logit)
        pq = pq/np.sum(pq)
    else:
        obj = lambda x: -np.sum(safe_q_logit*x) - 1/(1.-q)*(1.-np.sum(x**(2.-q)))
        const = ({'type':'eq', 'fun':lambda x:np.sum(x)-1.})
        bnds = [(0.,1.) for i in range(safe_q_logit.shape[0])]
        res = minimize(obj, x0=np.ones_like(safe_q_logit)/safe_q_logit.shape[0], constraints=const, bounds=bnds)
        maxq = -res.fun+max_q_logit
        pq = res.x
    return maxq, pq

def np_max_q(q_logits,q=1):
    maxq_list = []
    pq_list = []
    for q_logit in q_logits:
        maxq, pq = np_max_single_q(q_logit,q=q)
        pq_list.append(pq)
        maxq_list.append(maxq)
    return np.array(maxq_list), np.array(pq_list)
    
def np_q_entropy(p,q=1):
    q_ent_val = -np.sum(p*np_log_q(p,q=q),axis=1)
    q_ent_val[np.isnan(q_ent_val)] = 0
    return q_ent_val

def tf_exp_q(x,q):
    logit = 1+(1-q)*x
    safe_x = tf.maximum(logit,0)

    exp_q_x = tf.cond(tf.equal(q,1.),true_fn=lambda: tf.exp(x),false_fn=lambda: tf.pow(safe_x,1/(1-q)))
    return exp_q_x
    
def tf_log_q(x,q):
    safe_x = tf.maximum(x,1e-6)

    log_q_x = tf.cond(tf.equal(q,1.),true_fn=lambda: tf.log(safe_x),false_fn=lambda: (tf.pow(safe_x,1-q)-1)/(1-q))
    return log_q_x

def tf_tsallis_entropy(p,q):
    return tf.reduce_sum(-tf_log_q(p,q)*p,axis=1,keepdims=True)

def tf_tsallis_divergence_with_logits(p1,p2_q_logits,q):
    return tf.reduce_sum((tf_log_q(p1,q)-p2_q_logits)*p1,axis=1,keepdims=True)
    
def tf_tsallis_divergence(p1,p2,q):
    return tf.reduce_sum((tf_log_q(p1,q)-tf_log_q(p2,q))*p1,axis=1,keepdims=True)
    
def tf_tsallis_distance(p1,p2,q):
    return tf_tsallis_entropy((p1+p2)/2,q) - (tf_tsallis_entropy(p1,q)+tf_tsallis_entropy(p2,q))/2

def tf_random_q_normal(shape,q):
    dim = tf.cast(shape,tf.float32)[1]
    
    z = tf.random_normal(shape)
    z_square = tf.reduce_sum(tf.square(z),axis=1,keepdims=True)
    nu = 2.*(2.-q)/(1.-q)
    chi2_distribution = tf.contrib.distributions.Chi2(nu)
    a = chi2_distribution.sample(shape[0])
    a = tf.reshape(a,[-1,1])

    x = tf.cond(tf.equal(q,1.0),true_fn=lambda: z,false_fn=lambda: tf.sqrt((dim+2-dim*q)/(1-q))*z/tf.sqrt(a+z_square))
    return x

def tf_q_gaussian_distribution(x, mu, log_std, q):
    std = tf.exp(log_std)+1e-8
    dim = tf.cast(tf.shape(x),tf.float32)[1]
    gamma1 = tf.exp(tf.lgamma((2.-q)/(1.-q)))
    gamma2 = tf.exp(tf.lgamma((2.-q)/(1.-q)+dim/2.))
    K_q = tf.cond(tf.equal(q,1.0),
                  true_fn=lambda: tf.pow(tf.constant(np.sqrt(2.*np.pi),dtype=tf.float32),dim),
                  false_fn=lambda: tf.pow(tf.sqrt(((dim+4.)-(dim+2.)*q)/(1.-q)*tf.constant(np.pi,dtype=tf.float32)),dim)*gamma1/gamma2
                 )

    return tf_exp_q(-tf.reduce_sum(tf.square((x-mu)/std),axis=1)/((dim+4.)-(dim+2.)*q),q=q)/K_q/tf.reduce_prod(std,axis=1)

# Sampling q Gaussian using Box Muller Transform
#def tf_random_q_normal(shape,q):
#    q_prime = (3*q-1) / (1+q)
#    U1 = tf.random_uniform(shape)
#    U2 = tf.random_uniform(shape)
#    return tf.sqrt(-2.*tf_log_q(U1,q=q_prime))*tf.cos(tf.constant(2*np.pi)*U2)/tf.sqrt(q+1)

#def tf_q_gaussian_distribution(x, mu, log_invbeta, q):
#    beta = (tf.exp(log_invbeta)+1e-8)
#    gamma1 = tf.exp(tf.lgamma(1/(q-1)))
#    gamma2 = tf.exp(tf.lgamma((1+q)/2/(q-1)))

#    C_q = tf.cond(tf.equal(q,1),true_fn=lambda: tf.constant(1./np.sqrt(2.),dtype=tf.float32),false_fn=lambda: tf.sqrt(tf.constant(2.,dtype=tf.float32)) * gamma1 /gamma2 /tf.sqrt(q-1)/(1+q))
    
#    tf_pdf = tf_exp_q(-0.5*tf.reduce_sum(tf.square((x-mu) / beta),axis=1),q=q) /tf.reduce_prod(beta,axis=1)/C_q / tf.sqrt(tf.constant(2.*np.pi,dtype=tf.float32))
#    return tf_pdf 