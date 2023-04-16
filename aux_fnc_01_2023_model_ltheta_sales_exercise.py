# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   This version: 01/09/2023,           
   
   Auxiliary functions for 01_2023_model_ltheta_sales.py
                           01_2023_payoffs.py
   
   @author: Salomon
       A. Securtization Contracts (see paper)
       B. New functions for model LD.
           This version has 2 state varibles:
               1. ratio L/D ( D=1 normalized)
               2. theta: repayment rate on stock of loans.
       C. Adds dividends adjustment quadratic fnc for Risk neutral banks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
#import quantecon as qe 
import math
import collections
import pandas as pd
import numpy as np
import time
#for 3D graphs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#%%
#-----------------------------------------------------------------------------
#   Functions for Value Function Iteration
#-----------------------------------------------------------------------------
def ocost_fnc(L,nuL, otype):
    """
    Convex Origination cost
    """
    if otype == 'quad':
        return nuL*(L**2)

    elif otype == 'exp':
        return nuL*np.exp(L)

def div_cost(div,divbar,kappa):
    """ dividends quadratic adjustment cost fnc
    """
    return kappa*(div - divbar)**2    
    
def div_fnc(constant,divbar,kappa):
    #--------------------------------------------------------------------
    # Solves the quadratic equation derived from BC with Dividends Adjusment Costs
    # return the solution for the 
    # quadratic equation:
    # div + kappa*(div-divbar)**2 + constant = 0
    #--------------------------------------------------------------------   
    # the above equations becomes
    # div**2 + div *b1 + b2 = 0
    # where b1 and b2 are:
    
    # constant must be <0, so that implied dividends are >0
    if isinstance(constant, collections.Iterable):
        # iterable object
        constant[constant>=0] = np.nan
    else:
        # not iterable object
        if constant >= 0: constant = np.nan
    
    if kappa ==0:
        return -1*constant
    b1 = 1/kappa - 2*divbar
    b2 = divbar**2 + constant/kappa
    
    # It has 2 solutions:
    div1 = (- b1 + np.sqrt(b1**2 - 4*b2) )/2         #(positive root)
    #div2 = (- b1 - np.sqrt(b1**2 - 4*b2) )/2         #(negative root)
    # return the positive root
    return div1


def Udiv_fnc(div,sigma):
    # dividends fnc
    Udiv = (div**(1-sigma))/(1-sigma)
    
    return Udiv

def vfmax_fnc(mmu, vini, params, nwfix, lgrid):
    """
    ___________________________________________________________________________
    INPUTS
    mmu: prob(thetap = thetah |theta)
    vini: n_l x 2         continuation value
    nwfix: scalar         current bank's funds (cash holdings after sec)
    lgrid: n_l x 1        grid for lending choices
    params                model fixed parameters
    
    OUTPUTS
    Vf_mat[lp_idx, theta]: scalar, value of Vf() at lp = argmax Vf
    lp_idx: scalar, index of max l in lgrid.
    """
    if nwfix <= 0:
        return np.nan, np.nan
    # passign parameters
    betaB, loss,  kappa, divbar, nuL, sigma, psi, cost, otype  = params
    n_lgrid= len(lgrid)
    
    # compute dividends from nwfix (scalar) to all (lp)
    c_mat = lgrid + ocost_fnc(lgrid, nuL,otype) - 1 - nwfix
    div_mat = div_fnc(c_mat,divbar,kappa) 
    
    # check capital requirement constraint is satisfied
    #nw_tol = 1e-4
    #nw_const = nwfix - div_mat - div_cost(div_mat,divbar,kappa) -ocost_fnc(lgrid, nuL,otype) - psi*lgrid
    #const_violated = np.greater(-nw_const, 1e-4)
    
    # drop choices (lp,dp) that do NOT satisfy the nw constraint
    #div_mat[const_violated] = np.nan

    # compute Value function, vini[1,:,:] is for thetah
    vini = vini.reshape(2,n_lgrid)
    Vf_mat = div_mat + betaB*(mmu*vini[1,:] + (1 - mmu)*vini[0,:])
    #Vf_mat = Udiv_fnc(div_mat,sigma) + betaB*(mmu*vini[1,:] + (1 - mmu)*vini[0,:])

    
    #if all values of vf_mat are nans return nan
    if np.isnan(Vf_mat).all() == 1:
        return np.nan, np.nan
    # Find the index that maximize the value function
    lp_idx = np.where(Vf_mat == np.nanmax(Vf_mat))[0]
    return Vf_mat[lp_idx[0]], lp_idx[0]

#%%% NEW FUNCTION
def VFI_fnc(params, prices, grids, mu, tol, option):
    
    """
    Funtion performs a Value Function Iteration.
        
    """
    np.set_printoptions(precision = 6)
    # assign objects    
    lgrid, thgrid, Ptheta  = grids
    n_th = len(thgrid)
    n_lgrid = len(lgrid)
    betaB, loss, kappa, divbar, nuL, sigma, psi, cost, otype = params
    Rd, Rl, pi = prices
    
    # Step1. Compute returns, position 0 is low, 1 is high
    vspread = Rl-loss*pi
    Mtheta = [theta*(Rl - loss*pi) + loss*pi for theta in thgrid]
    Ml = Mtheta[0]
    Mh = Mtheta[1]
    # Step 2. Compute contracts
    if option=='NoSales':
        print('------------------------------------------------------')
        print('----------- No sales in Secondary Market--------------')
        print('Bank reputation (buyer) = '+ str(mu))
        print('------------------------------------------------------')
        Zcontract = np.zeros((2,2))
        qs_h  = 0
        qs_l  = 0
        zl = Zcontract[0,:]
        zh = Zcontract[1,:]
    elif option == 'CI':
        qs_h  = Mh
        qs_l  = Ml
        zh = np.array([1,qs_h])
        zl = np.array([1,qs_l])
        Zcontract = np.array([zl, zh]).reshape(2,2)
        EProfits = mu*(Mh - qs_h)*zh[0] + (1-mu)*(Ml - qs_l)*zl[0]
        print('------------------------------------------------------')
        print('--------Sales under Complete Information--------------')
        print('zh = (xh,qh) = '+'{:.3f}, {:.3f}'.format(zh[0],zh[1]))
        print('zl = (xl,ql) = '+'{:.3f}, {:.3f}'.format(zl[0],zl[1]))
        print('------------------------------------------------------')
        print('Loan sale prices :    (qs_h,qs_l) = '+'{:.3f}, {:.3f}'.format(qs_h,qs_l))
        print('Net return rates:    (Mh, Ml)= '+'{:.3f}, {:.3f}'.format(Mh,Ml))
        print('------------------------------------------------------')
        print(' Buyer Profit = ''{:.4f}'.format(EProfits) )    
        print('------------------------------------------------------')
        
    elif option=='AI':
        #---------------------------------------------------------------------------
        # Loan Sales Contract outcome
        #---------------------------------------------------------------------------
        #Mh  = thetah*(Rl-loss*pi)+loss*pi
        #Ml  = thetal*(Rl-loss*pi)+loss*pi
        # z_fnc returns contracts of the form: z=(x,t)
        #       where t=q*x - x*loss*pi,     t is the payment
        zh,zl, d, mut, eq = z_fnc(thgrid[1],thgrid[0],vspread,mu,cost)
        qs_h  = zh[1]/zh[0]+loss*pi
        qs_l  = zl[1]/zl[0]+loss*pi
        #rewrite contract z in terms of (x,q)
        zh = np.array([zh[0], qs_h])
        zl = np.array([zl[0], qs_l])        
        Zcontract = np.array([zl, zh]).reshape(2,2)
        qpool = mu*qs_h + (1-mu)*qs_l
        Eprofits = mu*(Mh - qs_h)*zh[0] + (1-mu)*(Ml - qs_l)*zl[0]
        ASd = (Mh-Ml)/cost 
        print('------------------------------------------------------')
        print('--------------- Equibrium Outcomes--------------------')
        print('Threshold: mu_tilde      = '+'{:.3f}'.format(mut))
        print('Adverse Selection, rho   = '+'{:.3f}'.format(ASd))
        print('Prob of thetah, mu       = '+str(mu))
        print('------------------------------------------------------')
        print('----> '+eq)
        print('------------------------------------------------------')
        print('zh = (xh,qh) = '+'{:.3f}, {:.3f}'.format(zh[0],zh[1]))
        print('zl = (xl,ql) = '+'{:.3f}, {:.3f}'.format(zl[0],zl[1]))
        print('------------------------------------------------------')
        print('Loan sale prices :    (qs_h,qs_l)      = '+'{:.3f}, {:.3f}'.format(qs_h,qs_l))
        print('Net return rates:    (Mh, Ml)          = '+'{:.3f}, {:.3f}'.format(Mh,Ml))
        print('Pooling price:          qpool(mu)      = '+'{:.3f}'.format(qpool))
        print('------------------------------------------------------')
        print(' Buyer Expected Profit = ''{:.4f}'.format(Eprofits) )    
        print('------------------------------------------------------')

    # Step 3. Compute cash holdings implied by states (l,Mtheta,Zcontract)
    nw_mat = np.array([lgrid*(qs_l*zl[0] + (1 - zl[0])*(Ml - cost)) - 1*Rd,
                       lgrid*(qs_h*zh[0] + (1 - zh[0])*(Mh - cost)) - 1*Rd]).flatten()

    # initial state nw CANNOT be less or equal to zero
    # replace with 0 all values nw_mat <=0, keep all others as implied by (l,d)
    nw_mat = nw_mat*np.logical_not(nw_mat<=0)
        
    # initial value for V
    vini = np.zeros((n_th*n_lgrid))  
    maxiter = 10000
    iter = 0
    stop = 1            
    while (stop>tol and iter < maxiter):
        vf_mat = np.array([vfmax_fnc(mu, vini, params, nw, lgrid)[0] \
                           for nw in nw_mat])
        diff = abs(vf_mat - vini)
        diff = diff[np.isfinite(diff)]
        stop = np.linalg.norm(diff,np.inf)                   
        vini = np.copy(vf_mat)
        iter+=1

    print('Convergence at iteration:', iter)
    print('Difference achieved:' + "{0:.6f}".format(stop))
    # after convergence compute policy functions
    vfinal, lp_idx = np.array([vfmax_fnc(mu, vini, params, nw, lgrid) \
                                       for nw in nw_mat]).T
    #store policy fnc values            
    pcf_l = idxtovalue(lp_idx,lgrid)
          
    # reshape into dim: nl x n_theta
    return vfinal.reshape(n_th,n_lgrid).T, \
            pcf_l.reshape(n_th,n_lgrid).T, \
            lp_idx.reshape(n_th,n_lgrid).T, \
            nw_mat.reshape(n_th,n_lgrid).T

#%%

def idxtovalue(lp_idx, lgrid):
    # returns array of the same shape of lp_idx with values
    # corresponding to lgrid, useful when lp_idx contains Nans
    pcf_l = []
    for idx in lp_idx:
        if np.isnan(idx):
            pcf_l.append(np.nan)
        else:
            pcf_l.append(lgrid[idx.astype(int)])
    return np.array(pcf_l)

def equity_const(nwfix, lp, dp, params):
    # returns the value of the net worth constraint for choice of (lp, dp)
    # given current nw
    
    # passign parameters
    betaB, loss,  kappa, divbar, nuL, sigma, psi, cost, otype  = params
        
    # compute dividends from nwfix (scalar) to all (lp, dp)
    c_mat = lp + ocost_fnc(lp, nuL,otype) - dp - nwfix
    div_mat = div_fnc(c_mat,divbar,kappa) 
    
    # check capital requirement constraint is satisfied 
    nw_const = nwfix - div_mat - div_cost(div_mat,divbar,kappa) - psi*lp
    #nw_tol = 1e-4
    #const_violated = np.greater(-nw_const, nw_tol)
    return nw_const

def stationary_mu_ld(Ild,Gld, Gtheta, P, mu0, tol):
    
    #################################################################
    #
    #INPUTS:
    #    P  : Nz x Nz          Transition matrix for theta.
    #    Gld : Nl x 1           Grid for l/d ratio.
    #    Ild : Nl x Nz          l/d pcy
    #    mu0 : Nl x Nz          Initial distribution
    #    
    #OUTPUTS:
    #    Mu: Nl x Nz            Stationary distribution matrix
    #
    #################################################################
    
    # Assign objects
    Nn    = len(Gld)
    Nz    = len(Gtheta)   
    
    #-----------------------------------------------------------------
    # Step 1: Construct indicators matrices
    #-----------------------------------------------------------------   
    # Indicator matrix, from being in state (ld,theta)--> (ldp,theta)
    B    = np.zeros((Nn,Nz,Nn)) 
    
    # Indicator matrices do not change in each iteration
    for i in range(Nn):
        B[:,:,i] = Ild == Gld[i]      

    #-----------------------------------------------------------------
    # Step 2: Iteration
    #-----------------------------------------------------------------
    C       = np.zeros((Nn,Nz))
    Mubar   = np.zeros((Nn,Nz))
    
    # Transition Matrix P: dim Nz x Nz, however...rows are the same,
    # since theta~ iid, rows are the same for every theta, i.e regardless of where you start
    # the probability of moving to the next theta is fixed and given by mu and (1-mu)    
    stop    = 1
    niter   = 0
    # Loop for iteration
    while stop>tol:
        for i in range(Nn):
            C[i,:] = np.sum(B[:,:,i]*mu0, axis=0) 
            Mubar  = np.dot(C,P)
            
        diff  = abs(Mubar- mu0)
        stop  = np.linalg.norm(diff,np.inf)
        temp  = mu0                   
        mu0   = Mubar
        Mubar = temp
        niter+=1        
        
    return Mubar

#%%



#for 3D graphs

def pcy_graph(Gx,Gy,pcyVar, option,legend,labelxy, title):
    """
    ======================
    3D surface (color map)
    ======================
    """
     
    # Create a new figure, plot into it, then close it so it never gets displayed
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X = Gx
    Y = Gy
    X, Y = np.meshgrid(X, Y)
    Z = pcyVar
    
    ax.set_xlabel(labelxy[0])
    ax.set_ylabel(labelxy[1])
    ax.set_zlabel(legend)

    if option==1:
        # Plot the surface.
        #surf = ax.plot_surface(X, Y, Z)
        
        graph = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(15))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(graph, shrink=0.5, aspect=5)
    else: 
        graph = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    
    plt.title(title)
        
    return graph



#%%

#-----------------------------------------------------------------------------
# Function for Contracts
#-----------------------------------------------------------------------------
def z_fnc(thetah,thetal,vspread,mu,cost):
    """
    This fnc characterizes the contract depending on value of returns M
    for each type. See Notes.
    INPUTS:
        cost:   Operational cost faced by banks
        psi:    Value of distribution of contracts in the Mixed Strategies Equilibrium
                (psi defined an specific contract)
        mu:     buyer's prior about bank's theta realization.
        thetah:     high type repayment rate
        thetal:     low type repayment rate
    
    OUTPUTS:
        Zh=(xh,th)
        Zl=(xl,tl)
        mut     threshold for reputation
        d       Adverse selection discount 
    """
    
    d       = (thetah-thetal)*vspread/cost
    mut     = d/(1+d)
    qpool   = (mu*thetah + (1-mu)*thetal)*vspread
    if mu < mut:
        
        xh      = 1/(1+d)
        th      = thetah*xh*vspread
        xl      = 1
        tl      = thetal*vspread
        eq      = "Pure Strategy Equilibrium Contract"
        zh      = xh,th
        zl      = xl,tl
        
    else:
        xl      = 1
        tl      = qpool - ((1-mu)*d)*(thetah-thetal)*vspread
        # Equation (12) in CSZ(2014)
        xh      = 1-((1-mu)/mu)*((d**-1 + d**-2)**-1)      
        th      = tl -(1-xh)*(thetal*vspread-cost)
        eq      = "Mixed Strategy Equilibrium Contract"
        zh      = xh,th
        zl      = xl,tl

    return zh,zl, d, mut, eq


def transitions(params, prices, grids, mu):
    """
    This function:
        1. solves for policy fnc, stationary distribution, contracts
        2. Calculate statistics
    
    """
    #-------------------------------------------------------------------------
    # 0. Assign parameters
    #-------------------------------------------------------------------------
    betaB, loss, kappa, divbar, nuL, sigma, psi, cost = params
    Rd, Rl, pi         = prices
    Gld, Gtheta, P     = grids
    thetal, thetah     = Gtheta
    N                  = len(Gld)
    Nz                 = len(Gtheta)
    #-------------------------------------------------------------------------
    # 1. Contracting outcomes
    #-------------------------------------------------------------------------
    #Mthetah = thetah*(Rl-loss*pi)+loss*pi
    #Mthetal = thetal*(Rl-loss*pi)+loss*pi
    vspread = Rl-loss*pi
    if mu ==0:
        contract = np.zeros((2,2))
    else:    
        zh,zl, d, mut, eq = z_fnc(thetah,thetal,vspread,mu,cost)
        contract = np.array([zl, zh]).reshape(2,2)
        qs_h  = zh[1]/zh[0]
        qs_l  = zl[1]/zl[0]    
    
    z_outcomes = qs_h, qs_l, d, mut
    #-------------------------------------------------------------------------
    # 2. Policy Functions
    #-------------------------------------------------------------------------
    tol     = 1e-5
    IV,Ild, Ildx, niter,stop = VFI_fnc(params, prices, grids, tol, mu)
    
    policy = IV, Ild, Ildx
    
    #-------------------------------------------------------------------------
    #                       Stationary Distribution
    #-------------------------------------------------------------------------

    # Initial distribution: uniformly distributed
    pdf0       = (1/(N*Nz))*np.ones((N,Nz))

    # Stationary Distribution
    pdf_loans  = stationary_mu_ld(Ild,Gld, Gtheta, P,pdf0, tol)
    
    # Aggregate Distribution across types
    pdf_tloans = np.sum(pdf_loans, axis=1)

    pdfs       = pdf_loans, pdf_tloans
    
    #-------------------------------------------------------------------------
    # 3. Statistics in Steady State for a given mu
    #-------------------------------------------------------------------------
    # Average Stock today
    Ll_stock = np.sum(pdf_loans[:,0]*Gld)
    Lh_stock = np.sum(pdf_loans[:,1]*Gld)
    L_stock = Ll_stock + Lh_stock
    
    # Total Expected sales in Steady State
    L_sold_exp = (1-mu)*zl[0]*Ll_stock + mu*zh[0]*Lh_stock

    # Average price
    
    # Stock of loans tomorrow
    #Llp_stock = np.sum(pdf_loans[:,0]*Ild[:,0])
    #Lhp_stock = np.sum(pdf_loans[:,1]*Ild[:,1])
    #Lp_stock  = Llp_stock + Lhp_stock
    
    # New originations
    #L_new   = Lp_stock - L_stock
    
    
    stats = L_stock, L_sold_exp
    
    return stats, z_outcomes, contract, policy, pdfs, 

def zstats_fnc(option, prices, thgrid, mu, loss, cost):
    """
    Computes statistics for each case in Secmkt=['NoSales', 'CI', 'AI']

    Parameters
    ----------
    option : array with options.  ['NoSales', 'CI', 'AI']
    prices : Rd, Rl, pi
    thgrid : [thetah, thetal].
    mu : prob of high type
    loss : recovery rate upon foreclosure
    pi : growth rate of house prices
    cost: bank's loan management cost
    
    Returns
    -------
    M_sprd, q_sprd, exsold, contract, qpool, ASD,  mut.
    """

    Rd, Rl, pi  = prices
    thetal, thetah= thgrid
    Mthetah     = thetah*(Rl-loss*pi)+loss*pi
    Mthetal     = thetal*(Rl-loss*pi)+loss*pi
    vspread     = Rl-loss*pi
        
    # Step1. Compute returns, position 0 is low, 1 is high
    Ml = Mthetal
    Mh = Mthetah
    # Step 2. Compute contracts    
    if option=='No Sales':
        print('------------------------------------------------------')
        print('----------- No sales in Secondary Market--------------')
        print('Bank reputation (buyer) = '+ str(mu))
        print('------------------------------------------------------')
        contract = np.zeros((2,2))
        qs_h  = 0
        qs_l  = 0
        zl = contract[0,:]
        zh = contract[1,:]
        qpool=0; ASD=0; mut=np.nan 
    elif option == 'CI':
        qs_h  = Mh
        qs_l  = Ml
        zh = np.array([1,qs_h])
        zl = np.array([1,qs_l])
        contract = np.array([zl, zh]).reshape(2,2)
        EProfits = mu*(Mh - qs_h)*zh[0] + (1-mu)*(Ml - qs_l)*zl[0]
        print('------------------------------------------------------')
        print('--------Sales under Complete Information--------------')
        print('zh = (xh,qh) = '+'{:.3f}, {:.3f}'.format(zh[0],zh[1]))
        print('zl = (xl,ql) = '+'{:.3f}, {:.3f}'.format(zl[0],zl[1]))
        print('------------------------------------------------------')
        print('Loan sale prices :    (qs_h,qs_l) = '+'{:.3f}, {:.3f}'.format(qs_h,qs_l))
        print('Net return rates:    (Mh, Ml)     = '+'{:.3f}, {:.3f}'.format(Mh,Ml))
        print('------------------------------------------------------')
        print(' Buyer Profit = ''{:.4f}'.format(EProfits) )    
        print('------------------------------------------------------')
        qpool=0; ASD=0; mut=np.nan
    elif option=='AI':
        #---------------------------------------------------------------------------
        # Loan Sales Contract outcome
        #---------------------------------------------------------------------------
        #Mh  = thetah*(Rl-loss*pi)+loss*pi
        #Ml  = thetal*(Rl-loss*pi)+loss*pi
        # z_fnc returns contracts of the form: z=(t,x)
        # where t=q*x - x*loss*pi,     t is the payment
        zh,zl, d, mut, eq = z_fnc(thgrid[1],thgrid[0],vspread,mu,cost)
        qs_h  = zh[1]/zh[0]+loss*pi
        qs_l  = zl[1]/zl[0]+loss*pi
        #rewrite contract in terms of (x,q)
        zh = np.array([zh[0], qs_h])
        zl = np.array([zl[0], qs_l])        
        contract = np.array([zl, zh]).reshape(2,2)
        qpool = mu*qs_h + (1-mu)*qs_l
        Eprofits = mu*(Mh - qs_h)*zh[0] + (1-mu)*(Ml - qs_l)*zl[0]
        ASD = (Mh-Ml)/cost 
        print('------------------------------------------------------')
        print('--------------- Equibrium Outcomes--------------------')
        print('Threshold: mu_tilde      = '+'{:.3f}'.format(mut))
        print('Adverse Selection, rho   = '+'{:.3f}'.format(ASD))
        print('Prob of thetah, mu       = '+str(mu))
        print('------------------------------------------------------')
        print('----> '+eq)
        print('------------------------------------------------------')
        print('zh = (xh,qh) = '+'{:.3f}, {:.3f}'.format(zh[0],zh[1]))
        print('zl = (xl,ql) = '+'{:.3f}, {:.3f}'.format(zl[0],zl[1]))
        print('------------------------------------------------------')
        print('Loan sale prices :    (qs_h,qs_l)      = '+'{:.3f}, {:.3f}'.format(qs_h,qs_l))
        print('Net return rates:    (Mh, Ml)          = '+'{:.3f}, {:.3f}'.format(Mh,Ml))
        print('Pooling price:          qpool(mu)      = '+'{:.3f}'.format(qpool))
        print('------------------------------------------------------')
        print(' Buyer Expected Profit = ''{:.4f}'.format(Eprofits) )    
        print('------------------------------------------------------')
    else:
        print('============================================================')
        print('Define Trading environment in Seconadry Market')
        print('Set \'option\' to: \'No Sales\', \'Sales CI\', \'Sales AI\' ')
        print('============================================================')    
    # Compute stats    
    exsold = zh[0]*mu + zl[0]*(1 - mu)      # Average fraction sold across all types
    M_sprd = Mthetah - Mthetal              # mortgage bond spread between types
    q_sprd = qs_h - qs_l                    # sec price spread between types
    Mtheta = [Mthetal, Mthetah]
    return Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut
    
#%%

def stats(pdf,x):
    aa= np.cumsum(pdf) 
    # Min of the distribution
    imin = np.min(np.where(aa>1e-6))
    x_min = x[imin]
    
    # Max of the distribution
    imax = np.min(np.where(aa>np.sum(pdf)-(1e-6)))
    x_max = x[imax]
    
    # mean
    x_mean = np.nansum(pdf*x)
    
    # variance, skewness and kurtosis
    temp2   = np.zeros(len(pdf))
    temp3   = np.zeros(len(pdf))
    temp4   = np.zeros(len(pdf))
    for i in range(len(pdf)):
        dist     = (x[i]-x_mean)
        temp2[i] = (dist**2)*pdf[i]      #for variance, 2nd central moment
        temp3[i] = (dist**3)*pdf[i]      #for skewness, 3rd central moment
        temp4[i] = (dist**4)*pdf[i]      #for kurtosis, 4th central moment           
    
    x_std  = np.nansum(temp2)**0.5
    #x_skew = np.sum(temp3)/(x_std**3)
    #x_kurt = np.sum(temp4)/(x_std**4)
    
    return x_mean, x_std, x_min, x_max

# Aggregate distributions
def agg_pdf(params, prices, grids, tol, mu, option):
    
    # declare params
    lgrid, thgrid, Ptheta = grids
    betaB, loss, kappa, divbar, nuL, sigma, psi, cost, otype = params

    # Obtain Vf and policy fnc
    print ("Working on : "+ option)
    start  = time.time() 
    vfinal, lpcf, lpcf_idx, nw_state = VFI_fnc(params, prices, grids, mu, tol, option)

    # Compute stationary distribution
    nlgrid = len(lgrid)
    ntheta = len(thgrid)
    pdf0     = (1/(nlgrid*ntheta))*np.ones((nlgrid,ntheta))
    l_pdf = stationary_mu_ld(lpcf_idx, np.arange(nlgrid), thgrid, Ptheta, pdf0, tol)
    
    # Compute stats
    divs = div_fnc(lpcf +ocost_fnc(lpcf,nuL,otype) - 1 - nw_state,divbar,kappa) 
    tot_lpdf    = np.sum(l_pdf, axis = 1)
    #tot_lcdf    = np.cumsum(tot_lpdf)   
    
    l_low  = stats(l_pdf[:,0], lgrid)
    l_high = stats(l_pdf[:,1], lgrid)
    l_tot  = stats(tot_lpdf, lgrid)      # total lending
    
    div_low = stats(l_pdf[:,0], divs[:,0]) 
    div_high= stats(l_pdf[:,1], divs[:,1]) 
    div_tot = np.array(div_low) + np.array(div_high)
    
    divasst_low  = div_low[0]/l_low[0]               # dividends to assets
    divasst_high = div_high[0]/l_high[0]             # dividends to assets
    divasst_tot  = div_tot[0]/l_tot[0]               # dividends to assets
    
    # Create DataFrame for pdf 
    lindex = lgrid /l_tot[0] - 1         # x-axis for plots as deviation from SS 
    df_pdf = pd.DataFrame(data=l_pdf, index=lindex)
    df_pdf.rename(columns = {0: "thetal", 1: "thetah"}, inplace=True)
    df_pdf["total"]=df_pdf.sum(axis=1)
    df_pdf["cdf"]=df_pdf.total.cumsum()
    #
    print ("elapsed time - sec =", ((time.time()-start)))
    
    return df_pdf, l_pdf, (divasst_low, divasst_high, divasst_tot), (l_low[0], l_high[0], l_tot[0])