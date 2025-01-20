import torch
import cooler
import click
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast
from importlib_resources import files
from torch.utils.data import DataLoader
from polaris.utils.util_loop import bedpewriter
from polaris.model.polarisnet import polarisnet
from polaris.utils.util_data import centerPredCoolDataset

@click.command()
@click.option('--batchsize', type=int, default=128, help='Batch size [128]')
@click.option('--cpu', type=bool, default=False, help='Use CPU [False]')
@click.option('--gpu', type=str, default=None, help='Comma-separated GPU indices [auto select]')
@click.option('--chrom', type=str, default=None, help='Comma separated chroms')
@click.option('-t', type=int, default=16, help='Number of cpu threads [16]')
@click.option('--max_distance', type=int, default=3000000, help='Max distance (bp) between contact pairs')
@click.option('--resol',type=int,default=5000,help ='Resolution')
@click.option('-i','--input', type=str,required=True,help='Hi-C contact map path')
@click.option('-o','--output', type=str,required=True,help='.bedpe file path to save loop candidates')
def score(batchsize, cpu, gpu, chrom, t, max_distance, resol, input, output, image=224):
    """Predict loop score for each pixel in the input contact map
    """
    print('Polaris Loop Prediction START :) ')
    
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
        for i in range(len(chrom)):
            if 'chr' not in chrom[i]:
                chrom[i] = f'chr{chrom[i]}'
        
    for rmchr in ['chrMT','MT','chrM','M','Y','chrY','X','chrX']: # 'Y','chrY','X','chrX'
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
        test_data = centerPredCoolDataset(coolfile,_chrom,max_distance_bin=max_distance//resol,w=image,step=center_size)
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

                loopwriter.write(_chrom,frag1,frag2,prob)              

if __name__ == '__main__':
    score()