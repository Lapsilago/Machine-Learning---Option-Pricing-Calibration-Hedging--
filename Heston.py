import numpy as np
from numpy import  sqrt
from numpy import zeros
from scipy import  pi, exp, real, log
from scipy.stats import norm
from scipy.integrate import quad,quadrature, trapz
import math
import random
from scipy.stats import norm,t
from scipy.optimize import least_squares,fminbound

class Heston(object):

    #Parameters of the class
    def __init__(self,S=1,r=0.025,q=0.0,kappa=2,vLong=0.05,sigma=0.3,v0=0.02,rho=-0.7):
        self.S = S 
        self.r = r
        self.q = q
        self.kappa = kappa
        self.vLong = vLong
        self.sigma = sigma
        self.v0 = v0
        self.rho = rho
        

############################# Heston Call/Put #####################################

    #Characteristic Function for the Heston model
    def heston_char_fkt(self,T,u):
        gamma = self.kappa - 1j*self.rho*self.sigma*u
        d = sqrt( gamma**2 + (self.sigma**2)*u*(u+1j) )
        g = (gamma - d)/(gamma + d)
        C = (self.kappa*self.vLong)/(self.sigma**2)*((gamma-d)*T-
          2*log((1 - g*exp(-d*T))/( 1 - g ) ))
        D = (gamma - d)/(self.sigma**2)*((1 - exp(-d*T))/
          (1 - g*exp(-d*T)))
        F = self.S*exp((self.r-self.q)*T)
        return exp(1j*u*log(F) + C + D*self.v0)


    #Heston Fundamental Transform
    def heston_trafo(self,T,u):
        F = self.S*exp((self.r-self.q)*T)
        return exp(-1j*u*log(F))* self.heston_char_fkt(T,u)
    
    
    #Heston Callprice by Lewis
    def heston_call_lewis(self,K,T):
        F = self.S*exp((self.r-self.q)*T)
        x = log(F/K)
        integrand = lambda k: real(exp(1j*k*x)/(k**2 + 1.0/4.0) * self.heston_trafo(T,k - 0.5*1j))
        integral = quad(integrand, 0, math.inf)[0]
        term1 = self.S * exp(-self.q*T)
        term2 = sqrt(self.S*K)/pi * exp(-(self.r+self.q)*T/2) * integral
        return  term1 - term2
    
    def heston_put_lewis(self,K,T):
        price = self.heston_call_lewis(K,T) - self.S + K *exp((self.r-self.q)*T)
        return price
    
    #Heston Callprice by Joshi-Yang
    # this is the Piterbarg version simply using sqrt(V0) as the BS vol
    # trapz provides a better way due to the exposition in Piterbarg but not implemented yet
    def heston_call_joshi(self,K,T):
        a = (self.v0 * T)**0.5
        d1 = (log(self.S /K) + ((self.r-self.q) + self.v0 / 2) * T) / a
        d2 = d1 - a
        BSCall = self.S * exp(-self.q*T) * norm.cdf(d1) - K * exp(-self.r*T) * norm.cdf(d2)
        F = self.S*exp((self.r-self.q)*T)
        x = log(F/K)
        integrand = lambda k: real(exp(1j*k*x)/(k**2 + 1.0/4.0) *
                                 (self.heston_trafo(T,k - 0.5*1j)- exp(-0.5*T*self.v0*(k**2 + 1.0/4.0))))
        integral = quadrature(integrand, 0, 150,tol=1e-6)[0]
        return (BSCall - sqrt(self.S*K)/pi * exp(-(self.r+self.q)*T/2) * integral)
    

    def heston_call_piterbarg(self,K,T):
        a = (self.v0 * T)**0.5
        d1 = (log(self.S /K) + ((self.r-self.q) + self.v0 / 2) * T) / a
        d2 = d1 - a
        BSCall = self.S * exp(-self.q*T) * norm.cdf(d1) - K * exp(-self.r*T) * norm.cdf(d2)
        logeps = log(0.00001)
        F = self.S*exp((self.r-self.q)*T)
        x = log(K/F)
        umax1 = fminbound(self.heston_f1,0,1000,args=(logeps,T,))
        umax2 = fminbound(self.heston_f2,0,1000,args=(logeps,T,))
        umax = max(umax1,umax2)
        X = np.linspace(0,umax,1000)
        integrand = lambda k: real(exp(-1j*k*x)/(k**2 + 0.25) *(exp(-0.5*T*self.v0*(k**2 + 0.25))-
                             self.heston_trafo(T,k - 0.5*1j)))
        integral = trapz(integrand(X),x=X)
        return (BSCall + sqrt(F*K)/pi * exp(-self.r*T) * integral)

    def heston_f1(self,u,logeps,T):
        return abs(-0.5* self.v0* T * u**2 - log(u) - logeps)
        
    def heston_f2(self,u,logeps,T):
        Cinf = (self.v0+self.kappa*self.vLong*T)/self.sigma*sqrt(1-self.rho**2)
        return abs(-Cinf*u - log(u) - logeps)   
                                                                     
                                                                     
    def heston_put_joshi(self,K,T):
        price = self.heston_call_joshi(K,T) - self.S + K *exp((self.r-self.q)*T)
        return price
    
    def heston_put_piterbarg(self,K,T):
        price = self.heston_call_piterbarg(K,T) - self.S + K *exp((self.r-self.q)*T)
        return price

################################ Heston Monte Carlo #####################################
    #Heston Monte Carlo
    
        
    def heston_qestep(self,Xt,Vt,deltat,gamma1,gamma2,psiC):    #QE discretization Agorithm for the Return Process {Xt+1}, yield one step ahead return conditional on Vt and Xt
        # new variance
        k1 = exp(-self.kappa*deltat)
        k2 = self.sigma**2 * k1 * (1-k1)/self.kappa
        k3 = exp(self.kappa*deltat)*0.5*k2*(1-k1)*self.vLong
        
        m = self.vLong +(Vt - self.vLong) * k1
        s2 = Vt * k2 + k3
        psi = s2/m**2
        if psi <= psiC:
            b2 = 2/psi-1 + (2/psi*(2/psi-1))**0.5 
            a = m/(1+b2)
            Zv = norm.ppf(random.random())
            Vnew = a*(Zv + b2**0.5)**2    #Non central Chi square variable aproximate sufficiently big value of Vt
        elif psi > psiC:
            p = (psi-1)/(psi+1)
            #beta = 2/ (m+m*psi)           #Function of Delta Dirac variable for sufficiently small value of Vt
            beta = (1-p)/m
            Uv = random.random()
            if Uv <=p:
                Vnew=0
            elif Uv> p:
                Vnew= log((1-p)/(1-Uv)) / beta
        
        # variables for the predictor-corrector step
        K0 = -1*(self.rho*self.kappa*self.vLong)*deltat/self.sigma 
        K1 = gamma1*deltat*(-0.5+(self.kappa*self.rho/self.sigma))-(self.rho/self.sigma)
        K2 = gamma2*deltat*(-0.5+(self.kappa*self.rho/self.sigma))+(self.rho/self.sigma)
        K3 = gamma1*deltat*(1-self.rho**2)
        K4 = gamma2*deltat*(1-self.rho**2)
        Zv = norm.ppf(random.random())              # Gaussian N(0,1)
        # predictor-corrector step
        Xnew = Xt + (self.r- self.q) *deltat + K0 + K1*Vt + K2*Vnew + ((K3*Vt+K4*Vnew)**0.5)*Zv
        return [Xnew,Vnew]


    def heston_qe(self,T,NSim,NT):
        gamma1 = 0.5 #averaging factors for the discretivazion of Xt
        gamma2 = 0.5 #averaging factors for the discretivazion of Xt
        psiC = 1.5   #Threshold for the initiation of the two aproximate distribution of V(t+1 | Vt)
        
        Nt = NT+1    # index 0 keeps the current values for S and V   
        deltat = T/NT
        
        # stores the paths
        pathS = zeros([NSim,Nt])
        pathV = zeros([NSim,Nt])
        pathS[:,0] = self.S
        pathV[:,0] = self.v0

        for i in range(NSim):
            Snew = np.log(self.S*exp(-self.q*T))
            Vnew = self.v0
            #print(Nt)
            for j in range(NT): 
                new = self.heston_qestep(Snew,Vnew,deltat,gamma1,gamma2,psiC) 
               
                Snew = new[0] 
                Vnew = new[1]

                pathS[i,j+1] = exp(Snew)
                pathV[i,j+1] = Vnew
    
        return pathS #, pathV]  #VT:Final Price  minPrice, maxPrice: max and min of the path, path: list of all Price value along the path