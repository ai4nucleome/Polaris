import torch
import click
import cooler
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast
from importlib_resources import files
from polaris.utils.util_loop import bedpewriter
from polaris.model.polarisnet import polarisnet
from scipy.sparse import coo_matrix
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def getLocal(mat, i, jj, w, N):
    if i >= 0 and jj >= 0 and i+w <= N and jj+w <= N:
        mat = mat[i:i+w,jj:jj+w].toarray()
        # print(f"global: {mat.shape}")
        return mat[None,...]
    # pad_width = ((up, down), (left, right))
    slice_pos = [[i, i+w], [jj, jj+w]]
    pad_width = [[0, 0], [0, 0]]
    if i < 0:
        pad_width[0][0] = -i
        slice_pos[0][0] = 0
    if jj < 0:
        pad_width[1][0] = -jj
        slice_pos[1][0] = 0
    if i+w > N:
        pad_width[0][1] = i+w-N
        slice_pos[0][1] = N
    if jj+w > N:
        pad_width[1][1] = jj+w-N
        slice_pos[1][1] = N
    _mat = mat[slice_pos[0][0]:slice_pos[0][1],slice_pos[1][0]:slice_pos[1][1]].toarray()
    padded_mat = np.pad(_mat, pad_width, mode='constant', constant_values=0)
    # print(f"global: {padded_mat.shape}",slice_pos, pad_width)
    return padded_mat[None,...]

def upperCoo2symm(row,col,data,N=None):
    # print(np.max(row),np.max(col),N)
    if N:
        shape=(N,N)
    else:
        shape=(row.max() + 1,col.max() + 1)

    sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
    symm = sparse_matrix + sparse_matrix.T
    diagVal = symm.diagonal(0)/2
    symm = symm.tocsr()
    symm.setdiag(diagVal)
    return symm

def processCoolFile(coolfile, cchrom, raw=False):
    extent = coolfile.extent(cchrom)
    N = extent[1] - extent[0]
    if raw:
        ccdata = coolfile.matrix(balance=False, sparse=True, as_pixels=True).fetch(cchrom)
        v='count'
    else:
        ccdata = coolfile.matrix(balance=True, sparse=True, as_pixels=True).fetch(cchrom)
        v='balanced'
    ccdata['bin1_id'] -= extent[0]
    ccdata['bin2_id'] -= extent[0]
        
    ccdata['distance'] = ccdata['bin2_id'] - ccdata['bin1_id']
    d_means = ccdata.groupby('distance')[v].transform('mean')
    ccdata[v] = ccdata[v].fillna(0)
    
    ccdata['oe'] = ccdata[v] / d_means
    ccdata['oe'] = ccdata['oe'].fillna(0)
    ccdata['oe'] = ccdata['oe'] / ccdata['oe'].max()
    oeMat = upperCoo2symm(ccdata['bin1_id'].ravel(), ccdata['bin2_id'].ravel(), ccdata['oe'].ravel(), N)
           
    return oeMat, N

@click.command()
@click.option('-b','--batchsize', type=int, default=128, help='Batch size [128]')
@click.option('-C','--cpu', type=bool, default=False, help='Use CPU [False]')
@click.option('-G','--gpu', type=str, default=None, help='Comma-separated GPU indices [auto select]')
@click.option('-c','--chrom', type=str, default=None, help='Comma separated chroms [all autosomes]')
@click.option('-t','--threshold', type=float, default=0.5, help='Loop Score Threshold [0.5]')
@click.option('-s','--sparsity', type=float, default=0.9, help='Allowed sparsity of submatrices [0.9]')
@click.option('-md','--max_distance', type=int, default=3000000, help='Max distance (bp) between contact pairs [3000000]')
@click.option('-r','--resol',type=int,default=5000,help ='Resolution [5000]')
@click.option('--raw',type=bool,default=False,help ='Raw matrix or balanced matrix')
@click.option('-i','--input', type=str,required=True,help='Hi-C contact map path')
@click.option('-o','--output', type=str,required=True,help='.bedpe file path to save loop candidates')
def scorelf(batchsize, cpu, gpu, chrom, threshold, sparsity, max_distance, resol, input, output, raw, image=224):
    """ *development* Score Pixels for Very Large mcool (>30GB) ...
    """
    print('\npolaris loop scorelf START :) ')

    center_size = image // 2
    start_idx = (image - center_size) // 2
    end_idx = (image + center_size) // 2
    slice_obj_pred = (slice(None), slice(None), slice(start_idx, end_idx), slice(start_idx, end_idx))
    slice_obj_coord = (slice(None), slice(start_idx, end_idx), slice(start_idx, end_idx))

    loopwriter = bedpewriter(output,resol,max_distance)
    
    if cpu:
        assert gpu is None, "\033[91m QAQ The CPU and GPU modes cannot be used simultaneously. Please check the command. \033[0m\n"
        gpu = ['None']
        device = torch.device("cpu")
        print('Using CPU mode... (This may take significantly longer than using GPU mode.)')
    else:
        if torch.cuda.is_available():
            if gpu is not None:
                print("Using the specified GPU: " + gpu)
                gpu=[int(i) for i in gpu.split(',')]
                device = torch.device(f"cuda:{gpu[0]}")
            else:
                gpuIdx = torch.cuda.current_device()
                device = torch.device(gpuIdx)
                print("Automatically selected GPU: " + str(gpuIdx))
                gpu=[gpu]
        else:
            device = torch.device("cpu")
            gpu = ['None']
            cpu = True
            print('GPU is not available!')
            print('Using CPU mode... (This may take significantly longer than using GPU mode.)')
           

    coolfile = cooler.Cooler(input + '::/resolutions/' + str(resol))
    modelstate = str(files('polaris').joinpath('model/sft_loop.pt'))
    _modelstate = torch.load(modelstate, map_location=device.type)
    parameters = _modelstate['parameters']

    if chrom is None:
        chrom =coolfile.chromnames
    else:
        chrom = chrom.split(',')
        
    # for rmchr in ['chrMT','MT','chrM','M','Y','chrY','X','chrX','chrW','W','chrZ','Z']: # 'Y','chrY','X','chrX'
    #     if rmchr in chrom:
    #         chrom.remove(rmchr)    

    print(f"Analysing chroms: {chrom}")
    
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
        
    badc=[]
    chrom_ = tqdm(chrom, dynamic_ncols=True)
    for _chrom in chrom_:
        chrom_.desc = f"[Analyzing {_chrom}]"

        oeMat, N = processCoolFile(coolfile, _chrom, raw)
        start_point = -(image - center_size) // 2
        joffset = np.repeat(np.linspace(0, image, image, endpoint=False, dtype=int)[np.newaxis, :], image, axis=0)
        ioffset = np.repeat(np.linspace(0, image, image, endpoint=False, dtype=int)[:, np.newaxis], image, axis=1)
        data, i_list, j_list = [], [], []
        count=0
        for i in range(start_point, N - image - start_point, center_size):
            for j in range(0, max_distance//resol, center_size):
                jj = j + i
                # if jj + w <= N and i + w <= N:
                _oeMat = getLocal(oeMat, i, jj, image, N)
                if np.sum(_oeMat == 0) <= (image*image*sparsity):
                    data.append(_oeMat)
                    i_list.append(i + ioffset)
                    j_list.append(jj + joffset)

            while len(data) >= batchsize or (i + center_size > N - image - start_point and len(data) > 0):
                count += len(data)
                
                bin_i = torch.tensor(np.stack(i_list[:batchsize], axis=0)).to(device)
                bin_j = torch.tensor(np.stack(j_list[:batchsize], axis=0)).to(device)
                targetX = torch.tensor(np.stack(data[:batchsize], axis=0)).to(device)
                bin_i = bin_i*resol
                bin_j = bin_j*resol

                data = data[batchsize:]
                i_list = i_list[batchsize:]
                j_list = j_list[batchsize:]

                # print(targetX.shape)
                # print(bin_i.shape)
                # print(bin_j.shape)

                with torch.no_grad():
                    with autocast():
                        pred = torch.sigmoid(model(targetX.float().to(device)))[slice_obj_pred].flatten()
                        loop = torch.nonzero(pred>threshold).flatten().cpu()
                        prob = pred[loop].cpu().numpy().flatten().tolist()
                        frag1 = bin_i[slice_obj_coord].flatten().cpu().numpy()[loop].flatten().tolist()
                        frag2 = bin_j[slice_obj_coord].flatten().cpu().numpy()[loop].flatten().tolist()

                    loopwriter.write(_chrom,frag1,frag2,prob)   
        if count == 0:
            badc.append(_chrom)
    
    if len(badc)==len(chrom):
        raise ValueError("polaris loop scorelf FAILED :( \nThe '-s' value needs to be increased for more sparse data.")
    else:
        print(f'\npolaris loop scorelf FINISHED :)\nLoopscore file saved at {output}')  
        if len(badc)>0:
            print(f"But the size of {badc} are too small or their contact matrix are too sparse.\nYou may need to check the data or run these chr respectively by increasing -s.")         
    
if __name__ == '__main__':
    scorelf()