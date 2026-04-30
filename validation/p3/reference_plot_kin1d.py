#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker,cm
import sys, getopt
#import getopt

#--------------------------------------------------------------------------------------------------
def plot_column_A(col):
    '''
    Plots for first column (col=0), for "general" fields, not specific to a particular ice category
    '''

    zet  = np.zeros((nt,nk), dtype='d')
    qitt = np.zeros((nt,nk), dtype='d')  #total (sum of all qitot(iice))
    nitt = np.zeros((nt,nk), dtype='d')  #total (sum of all nitot(iice))
    TT   = np.zeros((nt,nk), dtype='d')
    Frim = np.zeros((nt,nk), dtype='d')
    rhoi = np.zeros((nt,nk), dtype='d')
    Di   = np.zeros((nt,nk), dtype='d')
    Dhmax= np.zeros((nt,nk), dtype='d')
    qc   = np.zeros((nt,nk), dtype='d')
    qr   = np.zeros((nt,nk), dtype='d')
    Wup  = np.zeros((nt,nk), dtype='d')
    Rliq = np.zeros((nt,nk), dtype='d')   #2D array for convenience only (only value at k=nk is relevant)
    Rsol = np.zeros((nt,nk), dtype='d')   #2D array for convenience only (only value at k=nk is relevant)

# extract from 'data'.  Last index is the column, correponding to order in 'out_p3.dat'
    for t in range(nt):
        Wup[t,:]  = data[(t)*nk:(t+1)*nk, 1]
        Rliq[t,:] = data[(t)*nk:(t+1)*nk, 2]
        Rsol[t,:] = data[(t)*nk:(t+1)*nk, 3]
        zet[t,:]  = data[(t)*nk:(t+1)*nk, 4]
        TT[t,:]   = data[(t)*nk:(t+1)*nk, 5]
        qc[t,:]   = data[(t)*nk:(t+1)*nk, 6] *1.e+3   #convert to [mm]
        qr[t,:]   = data[(t)*nk:(t+1)*nk, 7] *1.e+3
        qitt[t,:] = data[(t)*nk:(t+1)*nk,10] *1.e+3
        nitt[t,:] = data[(t)*nk:(t+1)*nk,11]

    # apply mask:
    mask = qitt > 0.001
    Wup  = np.where(Wup>0., Wup, -1.)
    Rliq = np.where(Rliq>0., Rliq, -1.)
    Rsol = np.where(Rsol>0., Rsol, -1.)
    nitt   = np.log10(nitt+1.e-12)
    #zet  = np.where(mask, zet,  -99.)
    #Frim = np.where(mask, Frim, -99.)
    #Di   = np.where(mask, Di,   -99.)
    #Dhmax= np.where(mask, Dhmax,-99.)

    Qlev_small = 1.e-6

    cb_pad = 0.03
    lnwid  = 1.
    lncol  = 'black'

    im0 = ax[0,col].contourf(range(nt), z_agl, np.transpose(Wup), levels=[0.,1.,2.,3.,4.,5.], vmin=0., vmax=8., cmap='Greys')
    fig.colorbar(im0, ax=ax[0,col], shrink=0.8, pad=cb_pad)
    ax_b = ax[0,col].twinx()
    ax_b.plot(range(nt), Rliq, color='red')
    ax_b.plot(range(nt), Rsol, color='blue')
    ax_b.set_ylim([0.,50.])
    ax_b.tick_params(axis="y",direction="in", pad=-22)

    im1 = ax[1,col].contourf(range(nt), z_agl, np.transpose(zet), levels=levs_Z, colors=ccol_zet)
    fig.colorbar(im1, ax=ax[1,col], shrink=0.8, pad=cb_pad)

    im2 = ax[2,col].contourf(range(nt), z_agl, np.transpose(qc), levels=levs_Qc,   cmap='Greens')
    fig.colorbar(im2, ax=ax[2,col], shrink=0.8, pad=cb_pad)
    ax[2,col].contour( range(nt), z_agl, np.transpose(qc), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Green')
    ax[2,col].contour( range(nt), z_agl, np.transpose(qr), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Red')
    ax[2,col].contour( range(nt), z_agl, np.transpose(qitt), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Blue')

    im3 = ax[3,col].contourf(range(nt), z_agl, np.transpose(qr), levels=levs_Qr,   cmap='Reds')
    fig.colorbar(im3, ax=ax[3,col], shrink=0.8, pad=cb_pad)
    ax[3,col].contour( range(nt), z_agl, np.transpose(qr), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Red')
    ax[3,col].contour( range(nt), z_agl, np.transpose(qc), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Green')
    ax[3,col].contour( range(nt), z_agl, np.transpose(qitt), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Blue')

    im4 = ax[4,col].contourf(range(nt), z_agl, np.transpose(qitt), levels=levs_Q,   cmap='Blues')
    fig.colorbar(im4, ax=ax[4,col], shrink=0.8, pad=cb_pad)
    ax[4,col].contour( range(nt), z_agl, np.transpose(qc), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Green')
    ax[4,col].contour( range(nt), z_agl, np.transpose(qr), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Red')
    ax[4,col].contour( range(nt), z_agl, np.transpose(qitt), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors='Blue')

    im5 = ax[5,col].contourf(range(nt), z_agl, np.transpose(nitt), levels=levs_ni,   colors=ccol_ni)
    fig.colorbar(im5, ax=ax[5,col], shrink=0.8, pad=cb_pad)


    xloc = 93
    yloc = 12.5
    labsize = 10
    ax[0,col].text(xloc-15,yloc+1.3,'mm h$^{-1}$', size=labsize)
    ax[0,col].text(xloc,yloc,'m s$^{-1}$', size=labsize)
    ax[1,col].text(xloc,yloc,'dBZ',    size=labsize)
    ax[2,col].text(xloc,yloc,'g kg$^{-1}$', size=labsize)
    ax[3,col].text(xloc,yloc,'g kg$^{-1}$', size=labsize)
    ax[4,col].text(xloc,yloc,'g kg$^{-1}$', size=labsize)
    ax[5,col].text(xloc,yloc,'# m$^{-3}$',  size=labsize)
#   ax[3,col].text(xloc,yloc,'mm',     size=labsize)

    xloc = 5
    yloc = 10.5
    labsize = 14
    ax[0,col].text(xloc,yloc,'$w$', size=labsize)
    ax[0,col].text(xloc,yloc-1.8,'$R_{liq}$', size=labsize, color='red')
    ax[0,col].text(xloc,yloc-3.4,'$R_{sol}$', size=labsize, color='blue')
    ax[1,col].text(xloc,yloc,'$Z_e$',       size=labsize)
    ax[2,col].text(xloc,yloc,'$Q_c$',       size=labsize)
    ax[3,col].text(xloc,yloc,'$Q_r$',       size=labsize)
    ax[4,col].text(xloc,yloc,'$Q_{i,tot}$', size=labsize)
    ax[4,col].text(xloc,yloc-1.7,'(sum)', size=labsize-3)
    ax[5,col].text(xloc,yloc,'$log(N_{i,tot})$', size=labsize)
    ax[5,col].text(xloc,yloc-1.7,'(sum)', size=labsize-3)
    #ax[0,col].text(xloc,yloc-4.3,'(mm h$^{-1}$)', size=8)
    #ax[3,col].text(xloc,yloc,'$D_m$',       size=labsize)


    xaxislabel = 'Time (min)'
    yaxislabel = 'Height (km)'
    labsize = 12
    if col==0:
        ax[0,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[1,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[2,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[3,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[4,col].yaxis.set_label_text(yaxislabel, size=labsize)

    bottom_ax = ax.shape[0]-1
    ax[bottom_ax,col].xaxis.set_label_text(xaxislabel, size=labsize)
    ax[0,col].set_title(plotTitle[col], size=16, fontweight='bold', pad=30)


#--------------------------------------------------------------------------------------------------
def plot_column_B(col):
    '''
    Plots for second etc. (col 1 etc.), for fields from specific ice categories
    '''

    iice = col-1
    col_title = 'Ice Category '+str(iice+1)  # note: iice is Pythonized; iice=0 for ice category 1


   #these are for the ice category being plotted (hence, they are initialized as 2D arrays
    qit = np.zeros((nt,nk), dtype='d')
    qir = np.zeros((nt,nk), dtype='d')
    qil = np.zeros((nt,nk), dtype='d')
    nit = np.zeros((nt,nk), dtype='d')
    bir = np.zeros((nt,nk), dtype='d')
    zit = np.zeros((nt,nk), dtype='d')
    rhoi= np.zeros((nt,nk), dtype='d')
    dmi = np.zeros((nt,nk), dtype='d')

   #extract from 'data' for ice category iice.  Last index is the column, correponding to order in 'out_p3.dat'
    for t in range(nt):
        ind_ice = 15 + 8*iice  # to correspond with columns from kin1d
        qit[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice  ] *1.e+3  # convert to g kg-1
        qir[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+1] *1.e+3
        qil[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+2] *1.e+3
        nit[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+3]
        bir[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+4]
        zit[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+5]
        rhoi[t,:] = data[(t)*nk:(t+1)*nk, ind_ice+6]
        dmi[t,:]  = data[(t)*nk:(t+1)*nk, ind_ice+7] *1.e+3  # convert to mm

    Frim = np.zeros(qit.shape)
    Fliq = np.zeros(qit.shape)
    Frim = qir/(qit-qil+1.e-12) - 1.e-12
    Fliq = qil/(qit+1.e-12) - 1.e-12
    nit  = np.log10(nit+1.e-12)
    rhoi = rhoi - 1.e-12
    dmi  = dmi - 1.e-12

    Qlev_small = 1.e-6
    # apply mask:
#    mask = qit1 > 0.001

    cb_pad = 0.03
    lnwid  = 1.
    lncol  = 'black'

    im0 = ax[0,col].contourf(range(nt), z_agl, np.transpose(qit[:,:]), levels=levs_Q,   cmap='Blues') #
    fig.colorbar(im0, ax=ax[0,col], shrink=0.8, pad=cb_pad)
    ax[0,col].contour( range(nt), z_agl, np.transpose(qit[:,:]), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors=lncol)

    im1 = ax[1,col].contourf(range(nt), z_agl, np.transpose(Frim[:,:]), levels=levs_F,   cmap='RdYlBu_r')
    fig.colorbar(im1, ax=ax[1,col], shrink=0.8, pad=cb_pad)

    im2 = ax[2,col].contourf(range(nt), z_agl, np.transpose(Fliq[:,:]), levels=levs_F,   cmap='RdYlBu_r')
    fig.colorbar(im2, ax=ax[2,col], shrink=0.8, pad=cb_pad)
    ax[2,col].contour( range(nt), z_agl, np.transpose(qit[:,:]), levels=[Qlev_small], linewidths=lnwid, linestyles='dotted', colors=lncol)

    im3 = ax[3,col].contourf(range(nt), z_agl, np.transpose(nit[:,:]), levels=levs_ni,   colors=ccol_ni)
    fig.colorbar(im3, ax=ax[3,col], shrink=0.8, pad=cb_pad)

    im4 = ax[4,col].contourf(range(nt), z_agl, np.transpose(rhoi[:,:]), levels=levs_rho,   colors=ccol_rho)
    fig.colorbar(im4, ax=ax[4,col], shrink=0.8, pad=cb_pad)

    im5 = ax[5,col].contourf(range(nt), z_agl, np.transpose(dmi[:,:]), levels=levs_di,   colors=ccol_di)
    fig.colorbar(im5, ax=ax[5,col], shrink=0.8, pad=cb_pad)


    xloc = 93
    yloc = 12.5
    labsize = 10
    ax[0,col].text(xloc,yloc,'g kg$^{-1}$', size=labsize)
    ax[3,col].text(xloc,yloc,'# m$^{-3}$',  size=labsize)
    ax[4,col].text(xloc,yloc,'kg m$^{-3}$', size=labsize)
    ax[5,col].text(xloc,yloc,'mm',          size=labsize)

    xloc = 5
    yloc = 10.5
    labsize = 14
    ax[0,col].text(xloc,yloc,'$Q_{i,tot}$', size=labsize)
    ax[1,col].text(xloc,yloc,'$F_{i,rim}$', size=labsize)
    ax[2,col].text(xloc,yloc,'$F_{i,liq}$', size=labsize)
    ax[3,col].text(xloc,yloc,'$log(N_{i,tot})$', size=labsize)
    ax[4,col].text(xloc,yloc,'$\u03C1_i$', size=labsize)
    ax[5,col].text(xloc,yloc,'$D_{i}$', size=labsize)
    #ax[2,col].text(67,  yloc,'$R_{liq}$', size=labsize, color='red')
    #ax[3,col].text(67,  yloc-1.8,'$R_{sol}$', size=labsize, color='blue')
    #ax[4,col].text(64,  yloc-3.6,'(mm h$^{-1}$)', size=12)
    #ax[5,col].text(xloc,yloc,'$Q_c,Q_r$', size=labsize)


    xaxislabel = 'Time (min)'
    yaxislabel = 'Height (km)'
    labsize = 12
    if col==0:
        ax[0,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[1,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[2,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[3,col].yaxis.set_label_text(yaxislabel, size=labsize)
        ax[4,col].yaxis.set_label_text(yaxislabel, size=labsize)

    bottom_ax = ax.shape[0]-1
    ax[bottom_ax,col].xaxis.set_label_text(xaxislabel, size=labsize)
    ax[0,col].set_title(col_title, size=12)


#--------------------------------------------------------------------------------------------------
def get_args(argv):
    ''' Function to accept command line arguments to specify the input file, figure name, title, and nCat
    '''
    arg_infile  = ""
    arg_figname = ""
    arg_title   = ""
    arg_ncat    = ""
    arg_help    = "Arguments for {0}: -i <infile> -f <figname> -t <title> -n <ncat>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "h:i:f:t:n:", ["help=", "infile=", "figname=", "title=", "ncat="])

    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
        elif opt in ("-i", "--infile"):
            arg_infile = arg
        elif opt in ("-f", "--figname"):
            arg_figname = arg
        elif opt in ("-t", "--title"):
            arg_title = arg
        elif opt in ("-n", "--ncat"):
            arg_ncat = arg

    return arg_infile, arg_figname, arg_title, arg_ncat

#--------------------------------------------------------------------------------------------------

# override intput file, figure file name, and title (as specified above) if provided by command line arguments
infile      = ''
outfilename = ''
plotTitle   = ''
nCat        = ''
if __name__ == "__main__":
    infile, outfilename, plotTitle, nCat = get_args(sys.argv)

if infile == '':
    infile = '../out_p3.dat'
if plotTitle == '':
    plotTitle = 'TITLE'
if outfilename == '':
    outfilename = 'fig.png'
if nCat == '':
    nCat = 1
else:
    nCat = int(nCat)

#print('infile: ',infile)
#print('title:  ',plotTitle)
#print('figure: ',outfilename)

plotTitle = [plotTitle]

# read in data from cld1d; store as individual field arrays:
data = np.loadtxt(infile)

nk = 41                      # number of vertical levels
nt = int(data.shape[0]/nk)   # number of output times
#z_agl = data[0:nk,0]        # array of elevations AGL [m]  (column 0)
z_agl = data[0:nk,0]/1000.   # array of elevations AGL [km]  (column 0)


#--------  plotting intervals:
#levs_Z    = [-20., 0., 10., 20., 30., 40., 50., 60., 70., 80.]
levs_Z = [-30,-10,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
levs_Q  = [0.001, 0.1, 1., 2.5, 5.,7.5,10.,12.5,15.]
levs_Qc  = [0.001, 0.1, 1., 2.5, 5., 7.5, 10.]
levs_Qr  = levs_Qc
levs_F   = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
levs_ni  = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
levs_rho = [0.,100.,200.,300.,400.,500.,600.,700.,800.,900.,1000.]
levs_di  = [0., 1.,2.,3.,4.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,60.]

#levs_Vi   = [0.001, 0.2, 0.5, 1., 2., 3., 4., 5., 7., 10., 15., 20., 25.]
#levs_Di   = [1.e-3,1.,2.,3.,4.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.]
#levs_Dh   = levs_Di
#levs_rhoi = [0.001, 100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]
#levs_lami = [0.000, 25., 50., 75., 100., 200., 300., 400., 500.]
#levs_mui = [0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]

ccol_zet  = ['lightgrey','darkgrey','grey','lightskyblue','dodgerblue','blue','limegreen','forestgreen',
	     'darkgreen','yellow','orange','salmon','red','darkred','darkviolet','Indigo','darkblue','black']
ccol_ni   = ['blue','lightskyblue','green','limegreen','yellow','orange','red','darkviolet','Indigo']
ccol_qit  = ['lightgrey','grey','lightskyblue','dodgerblue','blue','limegreen','forestgreen','darkgreen','yellow','orange','salmon','red','violet','darkviolet','Indigo']
ccol_rho  = ['lightgrey','lightskyblue','dodgerblue','blue','limegreen','forestgreen','darkgreen','yellow','red','darkred']
ccol_di   = ccol_qit

#--------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(nrows=6, ncols=nCat+1, figsize=(7+5*nCat, 14), sharex=True)

plot_column_A(0)    #first column, "general" fields (not specific to a particular ice category)

# note: nCat is 1 by default. To indicate values greater than 1 (and to plot categories 2+), the
#       command line command must include the value of nCat after -n.  For example, to plot
#       a run with nCat=2, do '% python plot_ncol.py -n 2'.  Otherwise, only the first category
#       will be plotted.

for iice in range(nCat):
    plot_column_B(iice+1)    #second+ column(s), column of fields for each ice category

# adjust subplots
plt.subplots_adjust(hspace=0.10)
plt.subplots_adjust(wspace=0.10)

#plt.tight_layout()

#--------------------------
plt.savefig(outfilename, format='png')
#plt.show()
