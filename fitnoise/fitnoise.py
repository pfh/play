
"""

Two axes of time:
- execution sequence
- specialization sequence
  source -> bytecode -> ( theano expression )? -> value

A = LL'



x'Aix 
= x'(LL')ix 
= x'L'i Lix
= (Li x)' Lix

check L'i = Li' ... yes

"""

import warnings

have_theano = False
try:
    import theano
    from theano import tensor
    from theano import gradient
    try:
        from theano.tensor import nlinalg, slinalg
        have_theano = True
    except:
        warnings.warn("Couln't import linear algebra for theano, need a more recent version")
except:
    pass

if not have_theano:
    warnings.warn("Couldn't import theano, calculations may be slow")


import numpy, numpy.linalg, numpy.random
import scipy, scipy.optimize, scipy.stats, scipy.special


def is_theanic(x):
    return have_theano and isinstance(x, tensor._tensor_py_operators)

def as_tensor(x, dtype='float64'):
    if not isinstance(x, numpy.ndarray) and \
       not is_theanic(x):
        x = numpy.array(x)            

    if not x.dtype == dtype:
        x = x.astype(dtype)

    return x

    
def as_scalar(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 0
    return x


def as_vector(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 1
    return x


def as_matrix(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 2
    return x


def dot(a,b):
    a = as_tensor(a)
    b = as_tensor(b)
    return (
        tensor.dot if is_theanic(a) or is_theanic(b) 
        else numpy.dot
        )(a,b)


def take(a,indices,axis):
    if is_theanic(a) or is_theanic(indices):
        return tensor.take(a,indices,axis)
    else:
        return numpy.take(a,indices,axis)


def take2(a,i0,i1):
    return take(take(a,i0,0),i1,1)


def log(x):
    x = as_tensor(x)
    return (tensor.log if is_theanic(x) else numpy.log)(x)


#From Numerical Recipes in C
def lanczos_gammaln(x):
    x = as_tensor(x)
    y = x
    tmp = x + 5.5
    tmp = tmp - (x+0.5)*log(tmp)
    ser = 1.000000000190015
    for cof in (
        76.18009172947146, 
        -86.50532032941677, 
        24.01409824083091, 
        -1.231739572450155, 
        0.1208650973866179e-2, 
        -0.5395239384953e-5,
        ):
        y = y + 1
        ser = ser + cof / y
    return -tmp + log(2.5066282746310005*ser/x)


def gammaln(x):
    x = as_tensor(x)
    return lanczos_gammaln(x) if is_theanic(x) else scipy.special.gammaln(x)


#for i in xrange(1,10):
#    x = i
#    print numpy.exp(gammaln(x)), numpy.exp(lanczos_gammaln(x))
#import sys;sys.exit(0)


def inverse(A):
    A = as_matrix(A)
    return (nlinalg.matrix_inverse if is_theanic(A) else numpy.linalg.inv)(A)


def det(A):
    A = as_matrix(A)
    return (nlinalg.det if is_theanic(A) else numpy.linalg.det)(A)

#Doesn't have gradient
def cholesky(A):
    A = as_matrix(A)
    return (slinalg.cholesky if is_theanic(A) else numpy.linalg.cholesky)(A)


#Doesn't have gradient
def qr_complete(A):
    A = as_matrix(A)
    if is_theanic(A):
        return nlinalg.QRFull('complete')(A)
    else:
        return numpy.linalg.qr(A, mode="complete")



# Agnostic as to whether numpy or theano
# Mvnormal(dvector('mean'), dmatrix('covar'))
#
# TODO: maybe store covar as cholesky decomposition
#
class Mvnormal(object):
    def __init__(self, mean, covar):
        self.mean = as_vector(mean)
        self.covar = as_matrix(covar)


    # Not available for theano
    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result
    
    
    def log_density(self, x):
        x = as_vector(x)
        offset = x - self.mean
        n = self.mean.shape[0]
        
        #TODO: cleverer linear algebra
        return -0.5*(
            log(2*numpy.pi)*n
            + log(det(self.covar))
            + dot(offset, dot(inverse(self.covar), offset))
            )

    
    def p_value(self, x):
        x = as_vector(x)
        offset = x - self.mean
        df = self.covar.shape[0]
        q = dot(offset, dot(inverse(self.covar), offset))
        return stats.chi2.sf(q, df=df)
    
    
    # Not available for theano
    def random(self):
        A = cholesky(self.covar)
        return self.mean + dot(A.T, numpy.random.normal(size=len(self.mean)))
        

    def transformed(self, A):
        A = as_matrix(A)
        return Mvnormal(
            dot(A,self.mean),
            dot(dot(A,self.covar),A.T)
            )

            
    def shifted(self, x):
        x = as_matrix(x)
        return Mvnormal(self.mean+x, self.covar)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvnormal(take(self.mean,i,0), take2(self.covar,i,i))


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i, 'int32')
        i2 = as_vector(i, 'int32')
        x2 = as_vector(x2)
        
        mean1 = take(self.mean,i1,0)
        mean2 = take(self.mean,i2,0)
        offset2 = x2-mean2
        
        covar11 = take2(self.covar,i1,i1) 
        covar12 = take2(self.covar,i1,i2)
        covar21 = take2(self.covar,i2,i1)
        covar22 = take2(self.covar,i2,i2)
        covar22inv = inverse(covar22)
        covar12xcovar22inv = dot(covar12, covar22inv)
        
        return Mvnormal(
            mean1 + dot(covar12xcovar22inv,offset2),
            covar11 - dot(covar12xcovar22inv,covar21)
            )


class Mvt(object):
    def __init__(self, mean, covar, df):
        self.mean = as_vector(mean)
        self.covar = as_matrix(covar)
        self.df = as_scalar(df)


    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result


    def log_density(self, x):
        x = as_vector(x)
        offset = x - self.mean
        p = self.covar.shape[0]
        v = self.df
        return (
            gammaln(0.5*(v+p))
            - gammaln(0.5*v)
            - (0.5*p)*log(numpy.pi*v)
            - 0.5*log(det(self.covar))
            - (0.5*(v+p))*log(1+dot(offset, dot(inverse(self.covar), offset))/v)
            )
    
    
    def p_value(self, x):
        x = as_vector(x)
        offset = x - self.mean
        p = self.covar.shape[0]
        q = dot(offset, dot(inverse(self.covar), offset)) / p
        return stats.f.sf(q, dfn=p, dfd=self.df)


    def random(self):
        A = cholesky(self.covar)
        return (
            self.mean 
            + dot(A.T,numpy.random.normal(size=len(self.mean)))
              * numpy.random.chisquare(self.df) 
            )
    
    
    def transformed(self, A):
        A = as_matrix(A)
        return Mvt(
            dot(A,self.mean),
            dot(dot(A,self.covar),A.T),
            self.df
            )

            
    def shifted(self, x):
        x = as_matrix(x)
        return Mvt(self.mean+x, self.covar, self.df)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvt(take(self.mean,i,0), take2(self.covar,i,i), self.df)


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i, 'int32')
        i2 = as_vector(i, 'int32')
        x2 = as_vector(x2)
        p2 = len(i2)
        
        mean1 = take(self.mean,i1,0)
        mean2 = take(self.mean,i2,0)
        offset2 = x2-mean2
        
        covar11 = take2(self.covar,i1,i1) 
        covar12 = take2(self.covar,i1,i2)
        covar21 = take2(self.covar,i2,i1)
        covar22 = take2(self.covar,i2,i2)
        covar22inv = inverse(covar22)
        covar12xcovar22inv = dot(covar12, covar22inv)
        
        df = self.df
        
        return Mvt(
            mean1 + dot(covar12xcovar22inv,offset2),
            (covar11 - dot(covar12xcovar22inv,covar21))
              * ((df + dot(offset2,dot(covar22inv,offset2))) / (df + p2)),
            df + p2
            )



class _object(object): pass

def fit_noise(y, design, get_dist, initial, use_theano=True):
    y = as_matrix(y)
    design = as_matrix(design)
    initial = as_vector(initial)
    
    assert y.shape[1] == design.shape[0]
    
    n = y.shape[0]
    m = y.shape[1]
    
    items = [ ]
    for row in xrange(n):
        item = _object()
        item.row = row
        item.retain = numpy.arange(m)[ 
            numpy.isfinite(y[row]) & get_dist(row,initial).good
            ]
        if len(item.retain) <= design.shape[1]: continue
        
        Q,R = qr_complete(design[item.retain])
        i2 = numpy.arange(design.shape[1],len(item.retain))        
        item.tQ2 = Q[:,i2].T
        item.z2 = dot(item.tQ2, y[row,item.retain])
        items.append(item)
    
    
    def score_row(row, retain, tQ2, z2, param):
        return -(
            get_dist(row, param)
            .marginal(retain)
            .transformed(tQ2)
            .log_density(z2)
            )
    
    # Non-theanic
    def score(param):
        return sum(
            score_row(item.row,item.retain,item.tQ2,item.z2,param)
            for item in items
            )

    if not use_theano:
        return scipy.optimize.fmin(score, initial)


    vrow = tensor.iscalar('row')
    vretain = tensor.ivector('retain')
    vtQ2 = tensor.dmatrix('tQ2')
    vz2 = tensor.dvector('z2')
    vparam = tensor.dvector('param')
    
    vvalue = score_row(vrow, vretain, vtQ2, vz2, vparam)
    vgradient = gradient.grad(vvalue, vparam)
    vhessian = gradient.hessian(vvalue, vparam)
    func = theano.function(
        [vrow,vretain,vtQ2,vz2,vparam], 
        [vvalue,vgradient,vhessian],
        on_unused_input='ignore',
        allow_input_downcast=True)

    #func = theano.function(
    #    [vrow,vretain,vtQ2,vz2,vparam], 
    #    vvalue,
    #    on_unused_input='ignore',
    #    allow_input_downcast=True)
    
    def value_gradient_hessian(param):
        total_value = 0
        total_gradient = numpy.zeros(len(param))
        total_hessian = numpy.zeros((len(param),len(param)))
        for item in items:
            this_value, this_gradient, this_hessian = func(
                item.row,item.retain,item.tQ2,item.z2,param
                )
            total_value += this_value
            total_gradient += this_gradient
            total_hessian += this_hessian
        return total_value, total_gradient, total_hessian
    
    param = initial
    for i in xrange(20):
        v,g,h = value_gradient_hessian(param)
        print param, v
        param = param - numpy.linalg.solve(h,g)
    print param
    
    return param

    #return scipy.optimize.fmin(score, initial)




#A = tensor.dmatrix("A")
#x = tensor.dvector("x")

#print theano.pp(gradient.grad( dot(x,dot(A,x)), x ))

#print Mvt_normal(tensor.dvector("mean"), tensor.dmatrix("covar"))

#A = [[1,2],[3,4]]
#
#print qr_complete(A)
#
#
#tA = tensor.matrix('A')
#
#qrtA = qr_complete(tA)
#print theano.function([tA],qrtA)(A)

dist = Mvnormal([5,5,5,5],numpy.identity(4))

data = numpy.array([ dist.random() for i in xrange(1000) ])
#print data

#import pylab
#pylab.plot(data[:,0],data[:,1],'.')
#pylab.show()
print data.shape

print fit_noise(data, [[1],[1],[1],[1]], 
    lambda i,p: Mvt([0,0,0,0],p[1:]*numpy.identity(4), p[0]), 
    [5.0, 0.5,0.5,0.5,0.5],
    use_theano=1)






