import sys
import torch
import cooler
import click
import numpy as np
import pandas as pd
from importlib_resources import files

from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from sklearn.neighbors import KDTree
from polaris.utils.util_loop import bedpewriter
from polaris.model.polarisnet import polarisnet
from polaris.utils.util_data import centerPredCoolDataset
# from polaris.utils.util_data_2_3MB import centerPredCoolDataset

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

def pool(data,dc,resol,mindelta,minscore,output,refine,radius):
    """call loop from loop candidates by clustering
    """
    
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
    
    
    
@click.command()
@click.option('--batchsize', type=int, default=128, help='batch size [128]')
@click.option('--cpu', type=bool, default=False, help='Use CPU [False]')
@click.option('--gpu', type=str, default=None, help='Comma-separated GPU indices [auto select]')
@click.option('--chrom', type=str, default=None, help='loop calling for comma separated chroms')
@click.option('-t', type=int, default=16, help='number of cpu threads; [16]')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('--resol',type=int,default=5000,help ='resolution')
@click.option('--modelstate',type=str,default=None,help='trained model')
@click.option('--dc', type=int, default=5, help='distance cutoff for local density calculation in terms of bin. [5]')
@click.option('--minscore', type=float,default=0.6, help='min loopScore [0.6]')
@click.option('--radius', type=int, default=2, help='radius for KDTree to remove outliers. Sparser data requires a larger radius. [2]')
@click.option('--mindelta', type=float, default=5, help='min distance allowed between two loops [5]')
@click.option('--refine',type=bool,default = True,help ='refine loops. Should always set as True [True]')
@click.option('-i','--input', type=str,required=True,help='Hi-C contact map path')
@click.option('-o','--output', type=str,required=True,help='.bedpe file path to save loops')
def pred(batchsize, cpu, gpu, chrom, t, max_distance, resol, modelstate, dc, minscore, radius, mindelta, refine, input, output, image=224, center_size=112):
    """Predict loops from Hi-C contact map
    """
    if modelstate is None:
        modelstate = str(files('polaris').joinpath('model/sft_loop.pt'))

    start_idx = (image - center_size) // 2
    end_idx = (image + center_size) // 2
    slice_obj_pred = (slice(None), slice(None), slice(start_idx, end_idx), slice(start_idx, end_idx))
    slice_obj_coord = (slice(None), slice(start_idx, end_idx), slice(start_idx, end_idx))
    
    results=[]
    
    if cpu:
        device = torch.device("cpu")
        print('using CPU ...')
    else:
        if torch.cuda.is_available():
            if gpu is not None:
                print('using cuda: '+ gpu)
                gpu=[int(i) for i in gpu.split(',')]
                device = torch.device(f"cuda:{gpu[0]}")
            else:
                gpuIdx = torch.cuda.current_device()
                device = torch.device(gpuIdx)
                print('use gping ' + "cuda:" + str(gpuIdx))
                gpu=[gpu]
        else:
            device = torch.device("cpu")
            print('GPU is not available, using CPU ...')

    coolfile = cooler.Cooler(input + '::/resolutions/' + str(resol))
    _modelstate = torch.load(modelstate, map_location=device.type)
    parameters = _modelstate['parameters']

    if chrom is None:
        chrom =coolfile.chromnames
    else:
        chrom = chrom.split(',')
        for i in range(len(chrom)):
            if 'chr' not in chrom[i]:
                chrom[i] = f'chr{chrom[i]}'
        
    for rmchr in ['chrMT','MT','chrM','M','Y','chrY',]: # 'Y','chrY','X','chrX'
        if rmchr in chrom:
            chrom.remove(rmchr)    
                  
    print(f"\nAnalysing chroms: {chrom}")
    
    model = polarisnet(
            image_size=parameters['image_size'], 
            in_channels=parameters['in_channels'], 
            out_channels=parameters['out_channels'],
            embed_dim=parameters['embed_dim'], 
            depths=parameters['depths'],
            channels=parameters['channels'], 
            num_heads=parameters['num_heads'], 
            drop=parameters['drop'], 
            drop_path=parameters['drop_path'], 
            pos_embed=parameters['pos_embed']
    ).to(device)
    model.load_state_dict(_modelstate['model_state_dict'])
    if not cpu and len(gpu) > 1:
        model = nn.DataParallel(model, device_ids=gpu) 
    model.eval()
        
    chrom = tqdm(chrom, dynamic_ncols=True)
    for _chrom in chrom:
        test_data = centerPredCoolDataset(coolfile,_chrom,max_distance_bin=max_distance//resol,step=center_size)
        test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=t,prefetch_factor=4,pin_memory=(gpu is not None))
        
        chrom.desc = f"[analyzing {_chrom}]"
              
        with torch.no_grad():
            for X in test_dataloader:
                bin_i,bin_j,targetX=X
                bin_i = bin_i*resol
                bin_j = bin_j*resol
                with autocast():
                    pred = torch.sigmoid(model(targetX.float().to(device)))[slice_obj_pred].flatten()
                    loop = torch.nonzero(pred>0.5).flatten().cpu()
                    prob = pred[loop].cpu().numpy().flatten().tolist()
                    frag1 = bin_i[slice_obj_coord].flatten().cpu().numpy()[loop].flatten().tolist()
                    frag2 = bin_j[slice_obj_coord].flatten().cpu().numpy()[loop].flatten().tolist()

                # Store the results in memory
                for i in range(len(frag1)):                    
                    if frag1[i] < frag2[i] and frag2[i]-frag1[i] > 11*resol and frag2[i]-frag1[i] < max_distance:
                        results.append([_chrom, frag1[i], frag1[i] + resol, 
                                        _chrom, frag2[i], frag2[i] + resol, 
                                        prob[i]])
    df = pd.DataFrame(results)
    pool(df,dc,resol,mindelta,minscore,output,refine,radius)

if __name__ == '__main__':
    pred()