import numpy as np
import click
import pandas as pd
from polaris.utils.util_bcooler import bcool
from matplotlib import pylab as plt
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('wr',["w", "r"], N=256)

def p2LL(x, cw=3):
    """
    P2LL for a peak.
    Parameters:
    x      : sqaure matrix, peak and its surrandings
    cw     : lower-left corner width
    """
    c = x.shape[0] // 2
    llcorner = x[-cw:, :cw].flatten()
    if sum(llcorner) == 0:
        return 0,np.sum(x[c,c]>x[c-1:c+2,c-1:c+2])
    return x[c, c] / (sum(llcorner) / len(llcorner)),np.sum(x[c,c]>x[c-1:c+2,c-1:c+2])

@click.command()
@click.option('-w', type=int, default=10, help="window size (bins): (2w+1)x(2w+1) [10]")
@click.option('--savefig', type=str, default=None, help="save pileup plot to file [FOCI_pileup.png]")
@click.option('--p2ll', type=bool, default=False, help="compute p2ll [False]")
@click.option('--mindistance', type=int, default=None, help="min distance (bins) to skip, only for bedpe foci [2w+1]")
@click.option('--maxdistance', type=int, default=1e9, help="min distance (bins) to skip , only for bedpe foci [1e9]")
@click.option('--resol', type=int, default=5000, help="resolution [5000]")
@click.option('--oe', type=bool, default=True, help="O/E normalized [True]")
@click.argument('foci', type=str,required=True)
@click.argument('mcool', type=str,required=True)
def pileup(w,savefig,p2ll,mindistance,resol,maxdistance,foci,mcool,oe):
    ''' 2D pileup contact maps around given foci

    \b
    FOCI format: bedpe file contains loops
    \f
    :param w:
    :param savefig:
    :param p2ll:
    :param mindistance:
    :param resol:
    :param maxdistance:
    :param foci:
    :param mcool:
    :param oe:
    :return:
    '''
    if mindistance is None:
        mindistance=2*w+1
    if savefig is None:
        savefig=foci+'_pileup.png'
    bcoolFile = bcool(mcool + '::/resolutions/' + str(resol))
    pileup=np.zeros((2 * w + 1, 2 * w + 1))
    if '.bedpe' in foci:
        filetype='bedpe'
    else:
        filetype = 'bed'
    if oe:
        oeType='oe'
    else:
        oeType='o'

    foci = pd.read_csv(foci,sep='\t',header=None)

    if filetype == 'bedpe':
        foci=foci[foci[4]-foci[1]>mindistance*resol]
        foci=foci[foci[4]-foci[1]<maxdistance*resol]


    chroms=list(set(foci[0]))


    n=0
    for chrom in chroms:
        fociChr=foci[foci[0]==chrom]
        X = list(fociChr[1])
        if filetype=='bedpe':
            Y = list(fociChr[4])
        else:
            Y=X.copy()
        bmatrix = bcoolFile.bchr(chrom,decoy=False)

        for x,y in zip(X,Y):
            mat,meta= bmatrix.square(x,y,w,oeType)
            pileup+=mat[0,:,:]
            n+=1
    pileup/=n
    plt.figure(figsize=(2, 2))
    plt.imshow(pileup,cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    if p2ll:
        plt.title('P2LL=' + "{:.2f}".format(p2LL(pileup)[0]), fontsize=12)
    plt.savefig(savefig,dpi=600)
