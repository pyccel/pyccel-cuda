#$ header A(double [:])
def A(x):
    y[0] = x[0] + 1.0
    return y

x     = ones(3, float)
b     = ones(3, float)
#tol   = 1.0e-9
#m     = 10
#r     = b - A(x)
#s     = r
#alpha = 0.5
#ss    = dot(r,r)
#for i in range(1,m):
#    u     = 2.0*x
#    l     = dot(s,u)
#    ll    = dot(s,r)
#    alpha = ll/l
#    x  = x + alpha*s
#    r  = b - 2.0*x
#    ss = dot(r,r)
#    l  = dot(s,u)
#    ll = dot(r,u)
#    beta = -ll/l
#    s = r + beta*s
#print(x,i)
