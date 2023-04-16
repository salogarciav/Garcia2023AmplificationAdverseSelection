# -*- coding: utf-8 -*-
"""
#########################################################################
     Version: 01/13/2023
     
     @author: Salomon                              
     
     Simplest Banking Model.
     Note: This script is modified verison of 01_06_exercise_model_ltheta_sale.py
#########################################################################
"""

import numpy as np
import math 
import time
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt


from aux_fnc_01_2023_model_ltheta_sales_exercise import (z_fnc,
                                                         zstats_fnc,
                                                         ocost_fnc, 
                                                         div_cost,  
                                                         div_fnc, 
                                                         VFI_fnc, 
                                                         equity_const, 
                                                         stationary_mu_ld,
                                                         idxtovalue,
                                                         agg_pdf,
                                                         stats)
#%%
###############################################################################
# Defining Parameters
###############################################################################

# common parameters  
cost        = 0.012 #0.10                         # bank operating cost. See Table 2 Landvoigt
betaB       = 0.985 #0.95                         # discount factor banks. average real risk-free rate from a one-year Treasury bill: 1.56\% for 1990-2007.
loss        = 0.75  #0.70                         # Associated loss when banks sell a house, banks keep loss times value of the house

# adjustment cost func parameters
kappa       = 10   #12.50                         # scale parameter for div adjustment cost fcn
divbar      = 0.008                               # target dividends long-term

#specific parameters
thetah      = 0.906 #0.98                         # Based on avg default rates for securitized mortgage pools from McDash Analytics 2001-2007, see Adelino et al (2019)
thetal      = 0.841 #0.80                         # thetah = 1 - default_PSL (subprime), thetal = 1 - default_GSE (prime)

# Calibrated paramaters
sigma       = 2                                   # risk aversion parameter for banks
psi         = 0.08                                # Capital requirement

# Convex cost function parameters
nuL         = 0.00033 #0.00015                    #   --> using the exponential convex costs: oexp_fnc
otype       = 'exp' # types: 'exp','quad'         # quad: nuL=0.0045,         exp: nuL={0.00015 - 0.00030}

# PRICES 
Rl        = 1.0626#1/0.85                         # average real 30-year fixed mortgage rate including fees (1990-2007), as reported by Freddie Mac Primary Mortgage Market Survey 2018. 
Rd        = 1.005                                 # average real rate on 3-month cerfiticate deposits (1990-2007), from St Louis FRED.

# EXOGENOUS STATES
pi         = 1.05 #1.00                           # 0.05 = average growth rate of the House Price Index (FHFA) from 1990-2007
mu         = 0.7  #0.5                            # probability of theta = thetah

# EXOGENOUS STATES
# 1. Sequence of house prices, (in this version, house price sequence is known)
T       = 15
pi      = np.linspace(.75,1.35,T)
vspread = Rl-loss*pi
Mh      = thetah*vspread+loss*pi
Ml      = thetal*vspread+loss*pi
# 3. Create Transition Matrix
# Since theta~ iid, rows are the same for every theta, i.e regardless of where you start
# the probability of moving to the next theta is fixed and given by pi and (1-pi)
ntheta  = 2
Ptheta  = np.kron(np.ones((ntheta,1)),[1-mu, mu])
Prow    = np.array([1-mu, mu])

# INPUTS for system of equations
params  = betaB, loss, kappa, divbar, nuL, sigma, psi, cost 
prices  = Rd, Rl, pi
thgrid  = np.array([thetal, thetah ])

# pre-asigned 
zh_vec    = np.zeros((T,2))
zl_vec    = np.zeros((T,2))
d_vec     = np.zeros((T,1))
mut_vec   = np.zeros((T,1))
qsh_vec   = np.zeros((T,1))
qsl_vec   = np.zeros((T,1))
qpool_vec = np.zeros((T,1))
ny_h      = np.zeros((T,1))  
ny_l      = np.zeros((T,1))  

#%%
#---------------------------------------------------------------------------
# Loan Sales Contract outcome
#---------------------------------------------------------------------------
for i in range(T):
    zh,zl, d, mut, eq   = z_fnc(thetah,thetal,vspread[i],mu,cost)
    zh_vec[i,:]         = zh
    zl_vec[i,:]         = zl
    d_vec[i]            = d
    mut_vec[i]          = mut
    qsh_vec[i]          = (zh[1]+zh[0]*loss*pi[i])/zh[0]                    # price per-unit
    qsl_vec[i]          = (zl[1]+zl[0]*loss*pi[i])/zl[0]                    # price per-unit
    qpool_vec[i]       = (mu*thetah + (1-mu)*thetal )*vspread[i] +loss*pi[i]
    Eprofits            = mu*(zh[1]-thetah*zh[0]*vspread[i])+(1-mu)*(zl[1]-thetal*zl[0]*vspread[i])
    # Total payoffs: (1-x)l(M-c) + q*x*l
    ny_h[i]             = (1-zh[0])*(Mh[i]-cost)+ (qsh_vec[i])*zh[0]
    ny_l[i]             = (1-zl[0])*(Ml[i]-cost)+ (qsl_vec[i])*zl[0]
    print('Expected profits'+str(Eprofits))
    
# Total payoffs under complete info: l*M-Rd
# banks sell entire portoflio and avoid operational cost
ny_h_CI = Mh
ny_l_CI = Ml
ny_h_NS = Mh - cost
ny_l_NS = Ml - cost

# Expected Cash flow
Eny_NS = mu*Mh + (1-mu)*Ml - cost
Eny_CI = mu*Mh + (1-mu)*Ml
Eny_AI = mu*ny_h + (1-mu)*ny_l

#indicator for LCSO region
lcso = (mu<= mut_vec)
lcso = lcso.reshape(T)


#%% FIGURES FOR THE PAPER
# ---------------------------------------------------------- #
# FIGURE: Payoffs as function of pi (house price) shock
# ---------------------------------------------------------- #
ppi = pi - 1.05

fig = plt.figure(figsize=(7, 6))
fig.suptitle(r'Contracts $z = (x_h,q_h,x_l,q_l)$')
#ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# create axes spans where a certain condition is satisfied
trans1 = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
trans2 = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
ax1.set_title(r'Price, $q$')
ax1.fill_between(ppi, 0, 1, where = lcso, facecolor = 'gray', alpha = 0.25, transform = trans1)
ax1.scatter(ppi, qsh_vec, color='b', marker = 's', label=r'high type')
ax1.scatter(ppi, qsl_vec, color='r', marker = 'o', label=r'low type')
ax1.plot(ppi, Mh, color='b', label=r'$M_h$ return')
ax1.plot(ppi, Ml, linestyle='--', color='r', label=r'$M_l$ return')
ax1.set_xlim([min(ppi) - 0.003, max(ppi) - 0.015 ])
ax1.set_ylim([min(Ml) - 0.005, max(Mh) - 0.000])
ax1.set_ylabel(' $ q $', fontsize=12)
ax1.set_xlabel('$\pi$, change in house price', fontsize=12)
ax1.legend(loc='lower right', frameon=False, ncol=1, fontsize=8)

fig.subplots_adjust(wspace=.3)   # the amount of width reserved for space between subplots,
ax2.set_title(r'Fraction securitized, $x$')
ax2.fill_between(ppi, 0, 1, where = lcso, facecolor = 'gray', alpha = 0.25, transform = trans2)
ax2.scatter(ppi, zh_vec[:,0], color='b', marker = 's', label=None)
ax2.scatter(ppi, zl_vec[:,0], color='r', marker = 'o', label=None)
ax2.set_ylabel('$ x $', fontsize=12)
ax2.set_xlabel('$\pi$, change in house price', fontsize=12)
#ax2.legend(loc='lower right', frameon=False, ncol=1, fontsize=8)
ax2.set_xlim([min(ppi) - 0.005, max(ppi) - 0.015 ])
ax2.set_ylim([0.1, 1 + 0.03])

plt.savefig('fig4_z_contract_payoffs_v0404.pdf', format='pdf', dpi=1000)
plt.show()


# ---------------------------------------------------------- #
# FIGURE: Cash holdings after trading in Secondary Market
# ---------------------------------------------------------- #

fig1 = plt.figure(figsize=(6, 5))
#fig1.suptitle(r'Liquid funds after securitization')
ax = fig1.add_subplot(111) 
# create axes spans where a certain condition is satisfied
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.fill_between(ppi, 0, 1, where = lcso, facecolor = 'gray', alpha = 0.25, transform = trans)
ax.plot(ppi, ny_h_CI, color='b', linestyle = '-', label=r'high type - CI')
ax.scatter(ppi, ny_h, color='b', marker = 's', label=r'high type - AI')
ax.plot(ppi, ny_l_CI, color='r', linestyle = '--', label=r'low type - CI')
ax.scatter(ppi, ny_l, color='r', marker = 'o', label=r'low type - AI')
ax.set_ylabel(r'cash holdings, $\tilde{y}(\theta,z)$')
ax.set_xlabel('$\pi$, change in house price', fontsize=12)
ax.set_xlim([min(ppi) - 0.003, max(ppi) - 0.015 ])
ax.set_ylim([min(ny_l_CI) - 0.001, max(ny_h_CI) + 0.001])
ax.legend(loc='lower right', frameon=False, ncol=1, fontsize=8)
#
plt.savefig('fig5_cash_holdings_sec_v0404.pdf', format='pdf', dpi=1000)
plt.show()


#%% OTHER FIGURES OF INTEREST

# ---------------------------------------------------------- #
#        FIGURE: Payoffs as function of Mh-Ml Spread
# ---------------------------------------------------------- #
spread = 100*(Mh-Ml)


fig2 = plt.figure()
fig2.suptitle('Contracts z = (x,t)')
plt.subplot(121)
plt.title('Payment, t')
plt.scatter(spread, qsh_vec, color='r', marker = 'o', label=r'qh')
plt.scatter(spread, qsl_vec, color='b', marker = 's', label=r'ql')
plt.plot(spread, Mh, color='r', label=r'Mh')
plt.plot(spread, Ml, color='b', label=r'Ml')
plt.ylabel(' q- price ')
plt.xlabel('Mh-Ml spread (%)')
plt.legend(loc='best')
fig2.subplots_adjust(wspace=.3)   # the amount of width reserved for space between subplots,
plt.subplot(122)
plt.title('Fraction traded, x')
plt.scatter(spread, zh_vec[:,0], color='r', marker = 'o', label=r'xh')
plt.scatter(spread, zl_vec[:,0], color='b', marker = 's', label=r'xl')
plt.ylabel('x ')
plt.xlabel('Mh- Ml spread (%)')
plt.legend(loc='best')
#plt.savefig('Npcycts_g50.eps', format='eps', dpi=1000)
plt.show()


fig4 = plt.figure(figsize=(7, 6))
fig4.suptitle('Cash holdings after trade in Secondary Market')
plt.subplot(121)
plt.title('Low type')
plt.scatter(ppi, ny_l, color='r', label=r'AI')
plt.plot(ppi, ny_l_CI, color='r', label=r'CI')
plt.plot(ppi, ny_l_NS, color='r', linestyle='--', label=r'No Sales')
plt.ylabel(' cash holdings per-loan ')
plt.xlabel('$\pi$, change in price of collateral')
plt.legend(loc='best')

fig4.subplots_adjust(wspace=.3)   # the amount of width reserved for space between subplots,
plt.subplot(122)
plt.title('High type')
plt.scatter(ppi, ny_h, color='b', label=r'AI')
plt.plot(ppi, ny_h_CI, color='b', label=r'CI')
plt.plot(ppi, ny_h_NS, color='b', linestyle='--', label=r'No Sales')
#plt.ylabel(' cash holdings per-loan ')
plt.xlabel('$\pi$, change in price of collateral')
#plt.legend(loc='best')
#plt.savefig('Npcycts_g50.eps', format='eps', dpi=1000)
plt.show()


#
Eny_AI = Eny_AI.reshape((T))
Eny_CI = Eny_CI.reshape((T))
Eny_NS = Eny_NS.reshape((T))


# growth rates
Enyg_AI = 100*np.log(Eny_AI[1:]/Eny_AI[0:-1])
Enyg_CI = 100*np.log(Eny_CI[1:]/Eny_CI[0:-1])
Enyg_NS = 100*np.log(Eny_NS[1:]/Eny_NS[0:-1])

fig6 = plt.figure(figsize=(7, 6))
fig6.suptitle('Cash holdings dynamics after sales')
plt.subplot(121)
plt.title('Expected cash holdings')
plt.scatter(ppi, Eny_AI, color='b', label=r'AI')
plt.plot(ppi, Eny_CI, color='g', label=r'CI')
plt.plot(ppi, Eny_NS, color='k', linestyle = '--', label=r'No Sales')
plt.ylabel('cash holdings per-loan')
plt.xlabel('$\pi$')
plt.legend(loc='best')
#%
plt.subplot(122)
plt.title('Growth rate of cash holdings')
plt.scatter(ppi[1:], Enyg_AI, color='b', label=r'AI')
plt.plot(ppi[1:], Enyg_CI, color='g', label=r'CI')
plt.plot(ppi[1:], Enyg_NS, color='k', linestyle = '--', label=r'No Sales')
plt.ylabel('growth in percentage')
plt.xlabel('$\pi$')
plt.legend(loc='best')
plt.show()


fig1 = plt.figure()
plt.title('cash holdings after trading in Secondary market')
plt.scatter(spread, ny_h, color='b', label=r'AI')
plt.plot(spread, ny_h_CI, color='b', label=r'CI- high type')
plt.scatter(spread, ny_l, color='r', label=r'AI')
plt.plot(spread, ny_l_CI, color='r', label=r'CI - low type')
plt.ylabel('cash holdings per-loan')
plt.xlabel('spread (%)')
plt.legend(loc='best')
#plt.savefig('Npcycts_g50.eps', format='eps', dpi=1000)
plt.show()

#%% Effects over aggregate supply
#---------------------------------------------------------------------------
# Computing a sequence for aggregate lending
#---------------------------------------------------------------------------
# Securitization Contract outcome
Secmkt = ['NoSales', 'CI', 'AI']
option = Secmkt[2] 
pi      = np.linspace(.95,1.20,T)


tol     = 1e-6
params  = betaB, loss, kappa, divbar, nuL, sigma, psi, cost, otype 

llow_AI = np.zeros(T)
lhigh_AI = np.zeros(T)
ltot_AI =np.zeros(T)
#
llow_CI = np.zeros(T)
lhigh_CI = np.zeros(T)
ltot_CI = np.zeros(T)
for i in range(T):
    # Prices
    prices      = Rd, Rl, pi[i]    
    
    # Grids. # grid for lending
    # lmin > Rd/[Ml - cost] necessary for lgrid to be well defined
    ld_min  = max(0,Rd/(1.0188591 - cost) )       # This restriction comes from BC 
    nlgrid  = 500                               # size of grid L/D                
    lgrid   = np.linspace(ld_min*2.0,4.5,nlgrid)
    grids   = lgrid, thgrid, Ptheta
    
    # Asymmetric Info economy
    lstats_AI = agg_pdf(params, prices, grids, tol, mu, Secmkt[2])[3]
    llow_AI[i] = lstats_AI[0]
    lhigh_AI[i] = lstats_AI[1]
    ltot_AI[i] = lstats_AI[2]
    # Complete Info economy
    lstats_CI = agg_pdf(params, prices, grids, tol, mu, Secmkt[1])[3]
    llow_CI[i] = lstats_CI[0]
    lhigh_CI[i] = lstats_CI[1]
    ltot_CI[i] = lstats_CI[2]
    
# growth rates
Eltotg_AI = 100*np.log(ltot_AI/ltot_AI[6])
Eltotg_CI = 100*np.log(ltot_CI/ltot_CI[6])

# growth rates by bank-type
llowg_AI = 100*np.log(llow_AI/llow_AI[6])
lhighg_AI = 100*np.log(lhigh_AI/lhigh_AI[6])
#
llowg_CI = 100*np.log(llow_CI/llow_CI[6])
lhighg_CI = 100*np.log(lhigh_CI/lhigh_CI[6])
ppi = pi - 1.05


fig6 = plt.figure(figsize=(16,16))
fig6.suptitle('Dynamics of Lending')
plt.subplot(221)
plt.title('Aggregate Volume Lending')
plt.plot(ppi, ltot_AI, color='b', label=r'AI')
plt.plot(ppi, ltot_CI, color='g', label=r'CI')
plt.ylabel('lending per-deposit')
plt.xlabel('$\pi$')
plt.legend(loc='best')
#%
plt.subplot(222)
plt.title('Volume of lending by types')
plt.plot(ppi, lhigh_AI, color='b', linestyle="-", marker="s",label=r'AI - high type')
plt.plot(ppi, lhigh_CI, color='b', linestyle="-", marker=None,label=r'CI - high type')
plt.plot(ppi, llow_AI, color='r', linestyle="--",marker="o",label=r'AI - low type')
plt.plot(ppi, llow_CI, color='r', linestyle="-.",marker=None,label=r'CI - low type')
#plt.plot(ppi, ltot_AI, color='g', linestyle="-", label=r'CI')
plt.xlim([min(ppi) + 0.005, max(ppi) - 0.015 ])
#plt.ylim([-30, 25 ])
plt.ylabel('growth in percentage')
plt.xlabel('$\pi$')
plt.legend(loc='best')
#%
plt.subplot(223)
plt.title('Growth rate of aggregate lending')
plt.plot(ppi, Eltotg_AI, color='b', linestyle="-.",label=r'AI')
plt.plot(ppi, Eltotg_CI, color='g', linestyle="-", label=r'CI')
plt.xlim([min(ppi) + 0.005, max(ppi) - 0.015 ])
plt.ylim([-30, 25 ])
plt.ylabel('growth in percentage')
plt.xlabel('$\pi$')
plt.legend(loc='best')
#
plt.subplot(224)
plt.title('Growth rate of lending by types')
plt.plot(ppi, lhighg_AI, color='b', linestyle="-",label=r'AI - high type')
plt.plot(ppi, lhighg_CI, color='b', linestyle="-.",label=r'CI - high type')
plt.plot(ppi, llowg_AI, color='r', linestyle="-",label=r'AI - low type')
plt.plot(ppi, llowg_CI, color='r', linestyle="--", marker=None,label=r'CI - low type')
plt.xlim([min(ppi) + 0.005, max(ppi) - 0.015 ])
plt.ylim([-30, 25 ])
plt.ylabel('growth in percentage')
plt.xlabel('$\pi$')
plt.legend(loc='best')

# ---------------------------------------------------------- #
#        FIGURE: Aggregate Credit
# ---------------------------------------------------------- #
ppi = pi - 1.057143
fig = plt.figure(figsize=(6,5))
plt.title('Growth Rate of Aggregate Credit')
plt.plot(ppi, Eltotg_AI, color='b', linestyle="-.",label=r'AI economy')
plt.plot(ppi, Eltotg_CI, color='g', linestyle="-", label=r'CI economy')
plt.xlim([min(ppi) - 0.018, max(ppi) - 0.018 ])
plt.ylim([-30, 25 ])
plt.ylabel('Percentage')
plt.xlabel('$\pi$, change in house price', fontsize=12)
plt.legend(loc='best', frameon=False, ncol=1, fontsize=8)
plt.savefig('fig_aggregatecredit', format='pdf', dpi=1000)
plt.show()

#%% Understanding the amplification effect

# Rate of change of differences in cash holdings across types
diff_AI = ny_h.reshape(T) - ny_l.reshape(T)
rate_AI = np.log(diff_AI[1:]/diff_AI[:-1])
diff_CI=ny_h_CI - ny_l_CI
rate_CI = np.log(diff_CI[1:]/diff_CI[:-1])

fig5 = plt.figure()
plt.title('Rate of change of differences in cash holdings between types')
plt.plot(ppi[1:], rate_CI, color='k', label=r'CI')
plt.plot(ppi[1:], rate_AI, color='g', linestyle='-.', label=r'AI')
plt.ylabel('rate of change')
plt.xlabel(r'$\pi$, change in the price of collateral')
plt.legend(loc='best')
plt.show
#%%
fig3 = plt.figure(figsize=(10, 4))
fig3.suptitle(r'Threshold $\mu$, and Adverse Selection discount')
plt.subplot(121)
plt.title(r'$\tilde{\mu}$, threshold')
plt.scatter(spread, mut_vec, color='b')
plt.plot(spread, mu*np.ones(T), '--')
plt.xlabel('spread (%)')
plt.legend(loc='best')
fig3.subplots_adjust(wspace=.3)   # the amount of width reserved for space between subplots,
plt.subplot(122)
plt.title('d, adverse selection dsct')
plt.scatter(spread, d_vec, color='r')
plt.xlabel('spread (%)')
plt.legend(loc='best')
#plt.savefig('Npcycts_g50.eps', format='eps', dpi=1000)
plt.show()