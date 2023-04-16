# -*- coding: utf-8 -*-
"""
#########################################################################
     Version: 01/09/2023    
     @author: Salomon                               
       
     Banking Model.
     Origination stage: 2 states (l, theta)
     Secondary Markets: Static adverse selection as CSZ(2014)
     
  Notes:
      There are 2 types of decreasing convex funtions available:
             Quadractic
             Exponential
     Note: This script is revised verison of 01_06_exercise_model_ltheta_sale.py
             
#########################################################################
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd


from aux_fnc_01_2023_model_ltheta_sales_exercise import (z_fnc,
                                                         zstats_fnc,
                                                         ocost_fnc, 
                                                         div_cost,  
                                                         div_fnc, 
                                                         VFI_fnc, 
                                                         equity_const, 
                                                         stationary_mu_ld,
                                                         agg_pdf,
                                                         idxtovalue,
                                                         stats)
mydir=r'G:\My Drive\RESEARCH\JMP2C\Revision1_JHE\CODES_2023'
#%%
###############################################################################
# Defining Parameters
###############################################################################

# common parametrs
cost        = 0.012                             # bank operating cost. See Table 2 Landvoigt
betaB       = 0.985                             # discount factor banks. average real risk-free rate from a one-year Treasury bill: 1.56\% for 1990-2007.
loss        = 0.75                               # Associated loss when banks sell a house, banks keep loss times value of the house

# adjustment cost func parameters
kappa       = 10                                # scale parameter for div adjustment cost fcn
divbar      = 0.008                             # target dividends long-term

#specific parameters
thetah      = 0.906                              # Based on avg default rates for securitized mortgage pools from McDash Analytics 2001-2007, see Adelino et al (2019)
thetal      = 0.841                              # thetah = 1 - default_PSL (subprime), thetal = 1 - default_GSE (prime)

# Calibrated paramaters
sigma       = 2                                 # risk aversion parameter for banks
psi         = 0.08                               # Capital requirement

# Convex cost function parameters
nuL         = 0.00033                          #   --> using the exponential convex costs: oexp_fnc
otype       = 'exp' # types: 'exp','quad'            # quad: nuL=0.0045,         exp: nuL={0.00015 - 0.00030}

# PRICES 
Rl        = 1.0626                              # average real 30-year fixed mortgage rate including fees (1990-2007), as reported by Freddie Mac Primary Mortgage Market Survey 2018. 
Rd        = 1.005                               # average real rate on 3-month cerfiticate deposits (1990-2007), from St Louis FRED.

# EXOGENOUS STATES
pi         = 1.05                           # 0.05 = average growth rate of the House Price Index (FHFA) from 1990-2007
mu         = 0.7                            # probability of theta = thetah
# Create Transition Matrix: Ptheta
# Since theta~ iid, rows are the same for every theta, i.e regardless of where you start
# the probability of moving to the next theta is fixed and given by pi and (1-pi)
ntheta      =  2
Ptheta      = np.kron(np.ones((ntheta,1)),[1-mu, mu])
Prow        = np.array([1-mu, mu])

# INPUTS for system of equations
prices      = Rd, Rl, pi
thgrid      = np.array([thetal, thetah ])

#%%
#---------------------------------------------------------------------------
# Securitization Contract outcome
#---------------------------------------------------------------------------
Secmkt = ['NoSales', 'CI', 'AI']
option = Secmkt[2] 

# contract stats
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
Ml = Mtheta[0]; Mh = Mtheta[1]
zl = contract[0,:]; zh=contract[1,:]

#%%
"""
#--------------- AI bsln ---------------#
prices = Rd, Rl, 1.05
option = Secmkt[2] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_AI_bsln  = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])
# AI pi+
prices = Rd, Rl, 1.05 + 0.07
option = Secmkt[2] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_AI_pi_up = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])
# AI pi-
prices = Rd, Rl, 1.05 - 0.07
option = Secmkt[2] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_AI_pi_dn = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])
#--------------- CI bsln ---------------#
prices = Rd, Rl, 1.05
option = Secmkt[1] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_CI_bsln  = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])
# CI pi+
prices = Rd, Rl, 1.05 +0.07
option = Secmkt[1] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_CI_pi_up = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])
# CI pi-
prices = Rd, Rl, 1.05 - 0.07
option = Secmkt[1] 
Mtheta, M_sprd, q_sprd, exsold, contract, qpool, ASD, mut = zstats_fnc(option, prices, thgrid, mu, loss, cost)
zl = contract[0,:]; zh=contract[1,:]
st_CI_pi_dn = np.array([M_sprd, q_sprd, zh[1],zl[1], zh[0], exsold, ASD])

#----------------------------------------------------------#

print("-"*50)
title = "Statistics "
np.set_printoptions(precision=3)
tab1 = PrettyTable()
#tab1.set_style(int_format = "%.3f")
tab1.field_names = ["M sprd", "q sprd", "qh", "ql", "xh", "Ex", "ASD"]
tab1.add_row(["CI",  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
tab1.add_row(st_CI_bsln)
tab1.add_row(st_CI_pi_up)
tab1.add_row(st_CI_pi_dn)
tab1.add_row(["AI",  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
tab1.add_row(st_AI_bsln)
tab1.add_row(st_AI_pi_up)
tab1.add_row(st_AI_pi_dn)
tab1.float_format["M sprd"] = "2.4"
tab1.float_format["q sprd"] = "2.4"
tab1.float_format["qh"] = "2.4"
tab1.float_format["ql"] = "2.4"
tab1.float_format["xh"] = "2.4"
tab1.float_format["Ex"] = "2.4"
tab1.float_format["ASD"] = "2.4"
print(tab1.get_string(title=title))
print("-"*50)
#"""
#%% 
#-------------------------------------------------------------------------
#                      Grids 
#-------------------------------------------------------------------------
# grid for lending
# lmin > Rd/[Ml - cost] necessary for lgrid to be well defined
ld_min  = max(0,Rd/(Ml - cost) )       # This restriction comes from BC 
nlgrid  = 500                               # size of grid L/D                
lgrid   = np.linspace(ld_min*2.0,4.5,nlgrid)

grids   = lgrid, thgrid, Ptheta
tol     = 1e-6
params  = betaB, loss, kappa, divbar, nuL, sigma, psi, cost, otype 

#%%
#-------------------------------------------------------------------------
#                       Value Function Iteration
#-------------------------------------------------------------------------
start  = time.time() 
#IV,Ild, Ildx, niter,stop = VFI_vector(params, prices, grids, tol, mu, option)
vfinal, lpcf, lpcf_idx, nw_state = VFI_fnc(params, prices, grids, mu, tol, option)
#policy = Ildx.astype(int)
print ("elapsed time - sec =", ((time.time()-start)))
#
# check capital requirments constraint for policy fncs
const_value = equity_const(nw_state, lpcf, 1, params)
const_binding = (const_value<=0)
divs = div_fnc(lpcf +ocost_fnc(lpcf,nuL,otype) - 1 - nw_state,divbar,kappa) 
divcost = div_cost(divs, divbar, kappa) 
flow_funds= lpcf + ocost_fnc(lpcf,nuL,otype) + divs + divcost - nw_state - 1


# Figures
fig = plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Value function')
plt.plot(lgrid, vfinal[:,0], 'r', linestyle = '-', label = r'theta low')
plt.plot(lgrid, vfinal[:,1], 'b', linestyle = '--')
plt.legend(loc='best')
plt.xlabel('loan stock')

plt.subplot(232)
plt.title('Lending policy fnc')
plt.plot(lgrid, lpcf[:,0], 'r', linestyle = '-', label = r'theta low')
plt.plot(lgrid, lpcf[:,1], 'b', linestyle = '--')
plt.plot(lgrid,lgrid, 'k', linestyle = ':')
plt.legend(loc='best')
plt.xlabel('loan stock')

plt.subplot(233)
plt.title('Securitization policy fnc')
plt.plot(lgrid, lpcf[:,0]*zl[0], 'r', linestyle = '-', label = r'theta low')
plt.plot(lgrid, lpcf[:,1]*zh[0], 'b', linestyle = '--')
plt.plot(lgrid,lgrid, 'k', linestyle = ':')
plt.legend(loc='best')
plt.xlabel('stock of loans')

plt.subplot(234)
plt.title('dividends')
plt.plot(lgrid, divs[:,0], 'r', linestyle = '-', label = r'1 = low type')
plt.plot(lgrid, divs[:,1], 'b', linestyle = '--')
plt.legend(loc='best')
plt.xlabel('loan stock')

plt.subplot(235)
plt.title('dividends cost')
plt.plot(lgrid, divcost[:,0], 'r', linestyle = '-', label = r'1 = low type')
plt.plot(lgrid, divcost[:,1], 'b', linestyle = '--')
plt.legend(loc='best')
plt.xlabel('loan stock')

plt.subplot(236)
plt.title('Cash holdings, ny')
plt.plot(lgrid, nw_state[:,0], 'r', linestyle = '-', label = r'1 = low type')
plt.plot(lgrid, nw_state[:,1], 'b', linestyle = '--')
plt.legend(loc='best')
plt.xlabel('loan stock')
plt.subplots_adjust(wspace=0.3, hspace=0.35)
figname = mydir + '/figures/' + option + '_pcf_bsln.png'
fig.savefig(figname)
plt.close(fig)

#%
#-------------------------------------------------------------------------
#                       Stationary Distribution
#-------------------------------------------------------------------------
# Gues for initial distribution: uniformly distributed
pdf0     = (1/(nlgrid*ntheta))*np.ones((nlgrid,ntheta))

# Stationary Distribution
lpdf        = stationary_mu_ld(lpcf_idx, np.arange(nlgrid), thgrid, Ptheta, pdf0, tol)
# Note for myself:
#   - If sum(pdf)<1 make sure that the distribution is computed for 
#   the non "nan" values of the lending policy function (lpcf)
print('-'*50)
print('------- Check that pdf adds up to 1 -------')
print(' Sum(pdf)     = '+'{:.3f}'.format(np.sum(lpdf)))
print('-'*50)


#-------------------------------------------------------------------------
#                       Statistics
#-------------------------------------------------------------------------

tot_lpdf    = np.sum(lpdf, axis = 1)
tot_lcdf    = np.cumsum(tot_lpdf)   

l_low  = stats(lpdf[:,0], lgrid)
l_high = stats(lpdf[:,1], lgrid)
l_tot  = stats(tot_lpdf, lgrid)      # total lending

div_low = stats(lpdf[:,0], divs[:,0]) 
div_high= stats(lpdf[:,1], divs[:,1]) 
div_tot = np.array(div_low) + np.array(div_high)

divasst_low  = div_low[0]/l_low[0]               # dividends to assets
divasst_high = div_high[0]/l_high[0]             # dividends to assets
divasst_tot  = div_tot[0]/l_tot[0]               # dividends to assets

#-------------------------------------------------------------------------
#                       Figure: pdf
#-------------------------------------------------------------------------
# Create DataFrame for pdf 
lindex = lgrid #/l_tot[0] - 1         # x-axis for plots as deviation from SS 
lpdf_ = pd.DataFrame(data=lpdf, index=lindex)
lpdf_.rename(columns = {0: "thetal", 1: "thetah"}, inplace=True)
lpdf_["total"]=lpdf_.sum(axis=1)
lpdf_["cdf"]=lpdf_.total.cumsum()


# Distribution plots
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
lpdf_.thetah.rolling(5).mean().plot(ax=ax1, linestyle = '-', color = 'grey', alpha=1,label=r'$\theta_l$')
lpdf_.thetal.rolling(5).mean().plot(ax=ax1, linestyle = '-', color = 'blue', alpha=1,label=r'$\theta_h$')
ax1.set_ylim([0.0, 0.06])
ax1.title.set_text('Stationary pdf, by type')
ax1.set_ylabel('pdf', fontsize=10)
ax1.legend(loc='best', frameon=False, ncol=1)
#
lpdf_.total.rolling(5).mean().plot(ax=ax2, linestyle = '-', color = 'k', alpha=1, label=r'PDF')
ax2.title.set_text('Stationary PDF')
ax2.set_ylim([0.0, 0.06])
ax2.legend(loc='best', frameon=False, ncol=1)
ax2.set_xlabel(r'$l^\prime$ (lending) in deviation from SS', fontsize=10)
#
lpdf_.cdf.plot(ax=ax3, linestyle = '-', color = 'k', alpha=1, label=r'CDF')
ax3.title.set_text('Stationary CDF')
ax3.legend(loc='best', frameon=False, ncol=1)
#
figname = mydir + '/figures/' + option +'_dist_bsln.png'
fig.savefig(figname)
plt.close(fig)

#%%
#-------------------------------------------------------------------------
#         Statistics Table for Amplification effect
#-------------------------------------------------------------------------
from scipy.stats import kurtosis
#np.set_printoptions(precision=3)

pi = 1.05 - 0.07
prices      = Rd, Rl, pi
df_AI_pi_dn, pdf_AI_pi_dn, div_AI_pi_dn, lend_AI_pi_dn = agg_pdf(params, prices, grids, tol, mu, Secmkt[2])
df_CI_pi_dn, pdf_CI_pi_dn, div_CI_pi_dn, lend_CI_pi_dn = agg_pdf(params, prices, grids, tol, mu, Secmkt[1])


pi = 1.05 + 0.07
prices      = Rd, Rl, pi
df_AI_pi_up, pdf_AI_pi_up, div_AI_pi_up, lend_AI_pi_up = agg_pdf(params, prices, grids, tol, mu, Secmkt[2])
df_CI_pi_up, pdf_CI_pi_up, div_CI_pi_up, lend_CI_pi_up = agg_pdf(params, prices, grids, tol, mu, Secmkt[1])

pi = 1.05
prices      = Rd, Rl, pi
df_AI_bsln, pdf_AI_bsln, div_AI_bsln, lend_AI_pi_bsln = agg_pdf(params, prices, grids, tol, mu, Secmkt[2])
df_CI_bsln, pdf_CI_bsln, div_CI_bsln, lend_AI_pi_bsln = agg_pdf(params, prices, grids, tol, mu, Secmkt[1])
# No securitization
df_NS_bsln, pdf_NS_bsln, div_NS_bsln, lend_NS_bsln = agg_pdf(params, prices, grids, tol, mu, Secmkt[0])

# Moments = [mean, std, min, max]
#moments_nosales = stats(pdf_tnosales, lgrid)
mts_AI_pi_bsln  = np.concatenate(( stats(np.sum(pdf_AI_bsln, axis = 1), lgrid), div_AI_bsln),axis=0)
mts_AI_pi_up    = np.concatenate(( stats(np.sum(pdf_AI_pi_up, axis = 1), lgrid), div_AI_pi_up),axis=0)
mts_AI_pi_dn    = np.concatenate(( stats(np.sum(pdf_AI_pi_dn, axis = 1), lgrid), div_AI_pi_dn),axis=0)
mts_CI_pi_bsln  = np.concatenate(( stats(np.sum(pdf_CI_bsln, axis = 1), lgrid), div_CI_bsln),axis=0)
mts_CI_pi_up    = np.concatenate(( stats(np.sum(pdf_CI_pi_up, axis = 1), lgrid), div_CI_pi_up),axis=0)
mts_CI_pi_dn    = np.concatenate(( stats(np.sum(pdf_CI_pi_dn, axis = 1), lgrid), div_CI_pi_dn),axis=0)

#-----------------------------------------------------------------------#
# Table with Statistic
# Print table
#-----------------------------------------------------------------------#
print("-"*50)
title = "Statistics "
print(title)
np.set_printoptions(precision=3)
tab1 = PrettyTable()
#tab1.set_style(int_format = "%.3f")
tab1.field_names = ["l_mean", "l_std", "lmin", "lmax","dl_low","dl_high","dl_tot"]
tab1.add_row(["CI",np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
tab1.add_row( mts_CI_pi_bsln)
tab1.add_row(mts_CI_pi_up)
tab1.add_row(mts_CI_pi_dn  )
tab1.add_row(["AI",np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
tab1.add_row(mts_AI_pi_bsln)
tab1.add_row(mts_AI_pi_up )
tab1.add_row(mts_AI_pi_dn )
tab1.float_format["l_mean"] = "2.4"
tab1.float_format["l_std"] = "2.4"
tab1.float_format["lmin"] = "2.4"
tab1.float_format["lmax"] = "2.4"
tab1.float_format["dl_low"] = "2.4"
tab1.float_format["dl_high"] = "2.4"
tab1.float_format["dl_tot"] = "2.4"
print(tab1.get_string(title=title))
print("-"*50)



#%%
#######################################################################
# Figures comparing Stationary Distributions
#######################################################################
nlgrid =len(pdf0)


# ------------------------------------------------------------------------#
# Distribution plots: each ditribution wrt lgrid
# ------------------------------------------------------------------------#
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(lgrid, np.sum(pdf_AI_pi_dn, axis = 1),'grey',label=r' $\Delta \pi$ =-7%')
plt.plot(lgrid, np.sum(pdf_AI_bsln, axis = 1),'b', label=r'Baseline')
plt.plot(lgrid, np.sum(pdf_AI_pi_up, axis = 1), 'darkorange', label=r'$\Delta \pi$ =+7%')
#plt.xlim([1, 2.1])
plt.ylim([0, 0.10])
plt.xlabel('${l}^\prime$ (new originations) ', fontsize=11)
plt.legend(loc='upper right', frameon=False)
plt.title('Stationary PDF, Asymmetric Information');


plt.subplot(122)
plt.plot(lgrid, np.sum(pdf_CI_pi_dn, axis = 1), 'grey', label=r'$\Delta \pi$ =-7%')
plt.plot(lgrid, np.sum(pdf_CI_bsln, axis = 1), 'b', label=r'$Baseline')
plt.plot(lgrid, np.sum(pdf_CI_pi_up, axis = 1), 'darkorange', label=r'$\Delta \pi$ =+7%')
#plt.xlim([1, 2.1])
plt.ylim([0, 0.06])
plt.xlabel('${l}^\prime$ (new originations)', fontsize=11)
plt.legend(loc='upper right', frameon=False)
plt.title('Stationary PDF, Complete Information');

# ------------------------------------------------------------------------#
# Distribution plots: each ditribution normalized at their-own mean
# ------------------------------------------------------------------------#
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
df_AI_pi_up.total.rolling(10).mean().plot(ax=ax1, linestyle = '--', color = 'darkorange', alpha=1,label=r'$\Delta \pi$ =+7%')
df_AI_bsln.total.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'blue', alpha=1,label=r'Baseline')
df_AI_pi_dn.total.rolling(10).mean().plot(ax=ax1, linestyle = '-.', color = 'grey', alpha=1,label=r'$\Delta \pi$ =-7%')
ax1.set_ylim([0.0, 0.10])
ax1.set_xlim([-0.18, 0.18])
ax1.title.set_text('Stationary PDF, Asymmetric Information')
ax1.set_ylabel('pdf', fontsize=10)
ax1.legend(loc='best', frameon=False, ncol=1)
ax1.set_xlabel(r'New lending in deviations from SS', fontsize=10)

#
df_CI_pi_up.total.rolling(10).mean().plot(ax=ax2, linestyle = '--', color = 'darkorange', alpha=1,label=r'$\Delta \pi$ =+7%')
df_CI_bsln.total.rolling(10).mean().plot(ax=ax2, linestyle = '-', color = 'blue', alpha=1,label=r'Baseline')
df_CI_pi_dn.total.rolling(10).mean().plot(ax=ax2, linestyle = '-.', color = 'grey', alpha=1,label=r'$\Delta \pi$ =-7%')
ax2.title.set_text('Stationary PDF, Complete Information')
ax2.set_ylim([0.0, 0.05])
ax2.set_xlim([-0.18, 0.18])
ax2.legend(loc='best', frameon=False, ncol=1)
ax2.set_xlabel(r'New lending in deviations from SS', fontsize=10)
#
figname = mydir + '/figures/' + 'comparing_distributions.png'
fig.savefig(figname)
plt.close(fig)


# ------------------------------------------------------------------------#
#  Figure for JHE paper: pdfs with DataFrame
# Distribution plots: each ditribution normalized at the bsln mean
# ------------------------------------------------------------------------#
ltot_AI_bsln = stats(np.sum(pdf_AI_bsln,axis=1), lgrid)[0]
ltot_CI_bsln = stats(np.sum(pdf_CI_bsln,axis=1), lgrid)[0]
ltot_NS_bsln = stats(np.sum(pdf_NS_bsln,axis=1), lgrid)[0]
# Create DataFrame for pdf 
lindex = lgrid /ltot_AI_bsln - 1         # x-axis for plots as deviation from SS (baseline)
dfAI = pd.DataFrame(data = np.vstack((np.sum(pdf_AI_bsln,axis=1),
                           np.sum(pdf_AI_pi_up,axis=1),
                           np.sum(pdf_AI_pi_dn,axis=1))).T, 
                            index=lindex, columns=["bsln", "pi_up", "pi_dn"])
dfAI["CDF"] = dfAI.bsln.cumsum()

# Create DataFrame for pdf 
lindex = lgrid /ltot_CI_bsln - 1         # x-axis for plots as deviation from SS (baseline)
dfCI = pd.DataFrame(data = np.vstack((np.sum(pdf_CI_bsln,axis=1),
                           np.sum(pdf_CI_pi_up,axis=1),
                           np.sum(pdf_CI_pi_dn,axis=1))).T, 
                            index=lindex, columns=["bsln", "pi_up", "pi_dn"])
dfCI["CDF"] = dfCI.bsln.cumsum()




# Distribution plots - Figure 4
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
dfAI.pi_up.rolling(10).mean().plot(ax=ax1, linestyle = '--', color = 'darkorange', alpha=1,label=r'$\Delta \pi$ =+7%')
dfAI.bsln.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'blue', alpha=1,label=r'Baseline')
dfAI.pi_dn.rolling(10).mean().plot(ax=ax1, linestyle = '-.', color = 'grey', alpha=1,label=r'$\Delta \pi$ =-7%')
ax1.set_ylim([0.0, 0.071])
ax1.set_xlim([-0.3, 0.3])
ax1.title.set_text('Stationary PDF, Asymmetric Information')
ax1.set_ylabel('pdf', fontsize=10)
ax1.legend(loc='best', frameon=False, ncol=1)
ax1.set_xlabel(r'Lending volume, deviations from baseline', fontsize=10)

#
dfCI.pi_up.rolling(10).mean().plot(ax=ax2, linestyle = '--', color = 'darkorange', alpha=1,label=r'$\Delta \pi$ =+7%')
dfCI.bsln.rolling(10).mean().plot(ax=ax2, linestyle = '-', color = 'blue', alpha=1,label=r'Baseline')
dfCI.pi_dn.rolling(10).mean().plot(ax=ax2, linestyle = '-.', color = 'grey', alpha=1,label=r'$\Delta \pi$ =-7%')
ax2.title.set_text('Stationary PDF, Complete Information')
ax2.set_ylim([0.0, 0.071])
ax2.set_xlim([-0.3, 0.3])
ax2.legend(loc='best', frameon=False, ncol=1)
ax2.set_xlabel(r'Lending volume, deviations from baseline', fontsize=10)
#
figname = mydir + '/figures/' + 'comparing_distributions.pdf'
fig.savefig(figname, format='pdf', dpi=1000)
plt.close(fig)


#%%
#-------------------------------------------------------------------------
#                Section. Appendix - Figure 7 and 8
#-------------------------------------------------------------------------


# Create DataFrame for pdf - No Sales
lindex = lgrid         # x-axis for plots as deviation from SS (baseline)
df = pd.DataFrame(data = np.vstack((np.sum(pdf_AI_bsln,axis=1),
                           np.sum(pdf_CI_bsln,axis=1),
                           np.sum(pdf_NS_bsln,axis=1))).T, 
                            index=lindex, columns=["AIpdf", "CIpdf", "NSpdf"])
df["AIcdf"] = df.AIpdf.cumsum()
df["CIcdf"] = df.CIpdf.cumsum()
df["NScdf"] = df.NSpdf.cumsum()

# Distribution plots - Figure 7
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
df.NSpdf.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'black', alpha=1,label=r'No Sales')
df.CIpdf.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'green', alpha=1,label=r'Sales CI')
ax1.set_ylim([0.0, 0.06])
ax1.title.set_text('Stationary PDF over total loans')
ax1.set_ylabel('pdf', fontsize=10)
ax1.legend(loc='upper left', frameon=False, ncol=1)
#
df.NScdf.plot(ax=ax2, linestyle = '-', color = 'k', alpha=1, label=r'CDF')
df.CIcdf.plot(ax=ax2, linestyle = '-', color = 'g', alpha=1, label=r'CDF')
ax2.title.set_text('Stationary CDF')
#ax2.get_legend().remove()
#
figname = mydir + '/figures/' +'appendix_fig7.pdf'
fig.savefig(figname, format='pdf', dpi=1000)
plt.close(fig)

# Distribution plots  - Figure 8
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
df.NSpdf.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'black', alpha=1,label=r'No Sales')
df.CIpdf.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'green', alpha=1,label=r'Sales CI')
df.AIpdf.rolling(10).mean().plot(ax=ax1, linestyle = '-', color = 'blue', alpha=1,label=r'Sales AI')
ax1.set_ylim([0.0, 0.06])
ax1.title.set_text('Stationary PDF over total loans')
ax1.set_ylabel('pdf', fontsize=10)
ax1.legend(loc='upper left', frameon=False, ncol=1)
#
df.NScdf.plot(ax=ax2, linestyle = '-', color = 'k', alpha=1, label=r'CDF')
df.CIcdf.plot(ax=ax2, linestyle = '-', color = 'g', alpha=1, label=r'CDF')
df.AIcdf.plot(ax=ax2, linestyle = '-', color = 'b', alpha=1, label=r'CDF')
ax2.title.set_text('Stationary CDF')
#ax2.get_legend().remove()
#
figname = mydir + '/figures/' +'appendix_fig8.pdf'
fig.savefig(figname, format='pdf', dpi=1000)
plt.close(fig)