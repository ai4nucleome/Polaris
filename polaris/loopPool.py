import sys
import click
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
from tqdm import tqdm

def rhoDelta(data,resol,dc,radius): 
    
    pos = data[[1, 4]].to_numpy() // resol
    # remove singleton
    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=radius, return_distance=True)
    _l = []
    for v in NNindexes:
        _l.append(len(v))
    _l=np.asarray(_l)
    data = data[_l>5].reset_index(drop=True)
    
    # end of remove singleton
    pos = data[[1, 4]].to_numpy() // resol
    val = data[6].to_numpy()

    try:
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
        NNindexes, NNdists = posTree.query_radius(pos, r=dc, return_distance=True)
    except ValueError as e:
        if "Found array with 0 sample(s)" in str(e):
            print("#"*88,'\n#')
            print("#\033[91m Error!!! The data is too sparse. Please increase the value of: [radius]\033[0m\n#")
            print("#"*88,'\n')
            sys.exit(1)
        else:
            raise

    # calculate local density rho
    rhos = []
    for i in range(len(NNindexes)):
        # rhos.append(np.sum(np.exp(-(NNdists[i] / dc) ** 2)))
        rhos.append(np.dot(np.exp(-(NNdists[i] / dc) ** 2), val[NNindexes[i]]))
    rhos = np.asarray(rhos)

    # calculate delta_i, i.e. distance to nearest point with larger rho
    _r = 100
    _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
    deltas = rhos * 0
    LargerNei = rhos * 0 - 1
    for i in range(len(_indexes)):
        idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
        if idx.shape[0] == 0:
            deltas[i] = _dists[i][-1] + 1
        else:
            LargerNei[i] = _indexes[i][idx[0]]
            deltas[i] = _dists[i][idx[0]]
    failed = np.argwhere(LargerNei == -1).flatten()
    while len(failed) > 1 and _r < 100000:
        _r = _r * 10
        _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                deltas[failed[i]] = _dists[i][-1] + 1
            else:
                LargerNei[failed[i]] = _indexes[i][idx[0]]
                deltas[failed[i]] = _dists[i][idx[0]]
        failed = np.argwhere(LargerNei == -1).flatten()

    data['rhos']=rhos
    data['deltas']=deltas

    return data

@click.command()
@click.option('--dc', type=int, default=5, help='Distance cutoff for local density calculation in terms of bin. [5]')
@click.option('--minscore', type=float,default=0.6, help='min loopScore [0.6]')
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('--radius', type=int, default=2, help='Radius threshold to remove outliers. [2]')
@click.option('--mindelta', type=float, default=5, help='Min distance allowed between two loops [5]')
@click.option('-i','--candidates', type=str,required=True,help ='Loop candidates file path')
@click.option('-o','--output', type=str,required=True,help ='.bedpe file path to save loops')
def pool(dc,candidates,resol,mindelta,minscore,output,radius,refine=True):
    """Call loops from loop candidates by clustering
    """
    data = pd.read_csv(candidates, sep='\t', header=None)
    
    if data.shape[0] == 0:
        print("#"*88,'\n#')
        print("#\033[91m Error!!! The file is empty. Please check your file.\033[0m\n#")
        print("#"*88,'\n')
        sys.exit(1)
    data = data[data[6] > minscore].reset_index(drop=True)
    data = data[data[4] - data[1] > 11*resol].reset_index(drop=True)
    if data.shape[0] == 0:
        print("#"*88,'\n#')
        print("#\033[91m Error!!! The data is too sparse. Please decrease: [minscore] (minimum: 0.5).\033[0m\n#")
        print("#"*88,'\n')
        sys.exit(1)
    data[['rhos','deltas']]=0
    data=data.groupby([0]).apply(rhoDelta,resol=resol,dc=dc,radius=radius).reset_index(drop=True)
    minrho=0
    targetData=data.reset_index(drop=True)

    loopPds=[]
    chroms=tqdm(set(targetData[0]), dynamic_ncols=True)
    for chrom in chroms:
        chroms.desc = f"[Runing clustering on {chrom}]"
        data = targetData[targetData[0]==chrom].reset_index(drop=True)

        pos = data[[1, 4]].to_numpy() // resol
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')

        rhos = data['rhos'].to_numpy()
        deltas = data['deltas'].to_numpy()
        centroid = np.argwhere((rhos > minrho) & (deltas > mindelta)).flatten()

        # calculate delta_i, i.e. distance to nearest point with larger rho
        _r = 100
        _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
        LargerNei = rhos * 0 - 1
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                pass
            else:
                LargerNei[i] = _indexes[i][idx[0]]

        failed = np.argwhere(LargerNei == -1).flatten()
        while len(failed) > 1 and _r < 100000:
            _r = _r * 10
            _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
            for i in range(len(_indexes)):
                idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                if idx.shape[0] == 0:
                    pass
                else:
                    LargerNei[failed[i]] = _indexes[i][idx[0]]
            failed = np.argwhere(LargerNei == -1).flatten()

        # assign rest loci to loop clusters
        LargerNei = LargerNei.astype(int)
        label = LargerNei * 0 - 1
        for i in range(len(centroid)):
            label[centroid[i]] = i
        decreasingsortedIdxRhos = np.argsort(-rhos)
        for i in decreasingsortedIdxRhos:
            if label[i] == -1:
                label[i] = label[LargerNei[i]]

        # refine loop
        val = data[6].to_numpy()
        refinedLoop = []
        label = label.flatten()
        for l in set(label):
            idx = np.argwhere(label == l).flatten()
            if len(idx) > 0:
                refinedLoop.append(idx[np.argmax(val[idx])])
        if refine:
            loopPds.append(data.loc[refinedLoop])
        else:
            loopPds.append(data.loc[centroid])

    loopPd=pd.concat(loopPds).sort_values(6,ascending=False)
    loopPd[[1, 2, 4, 5]] = loopPd[[1, 2, 4, 5]].astype(int)
    loopPd[[0,1,2,3,4,5,6]].to_csv(output,sep='\t',header=False, index=False)
    print(len(loopPd),'loops saved to ',output)

if __name__ == '__main__':
    pool()
