import click
import cooler
import numpy as np
from tqdm import tqdm 
np.seterr(divide='ignore', invalid='ignore')

@click.command()
@click.option('-c','--chrom', type=str, default=None, help='Comma separated chroms [all autosomes]')
@click.option('-md','--mindis', type=int, default=0, help='Only count reads with genomic distance (in bins) greater than this value. [0]')
@click.option('-r','--resol',type=int,default=5000,help ='Resolution [5000]')
@click.option('-i','--input', type=str,required=True,help='mcool file path')
def depth(input, resol, mindis, chrom):
    """Calculate intra reads of mcool file
    """
       
    print(f'\npolaris util depth START :)')
    totals = 0
    C = cooler.Cooler(f"{input}::resolutions/{resol}")
    mindis = mindis // C.binsize

    if chrom is None:
        chrom =C.chromnames
    else:
        chrom = chrom.split(',')
    print(f"Calculating depth for {chrom}")
    
    chrom_ = tqdm(chrom, dynamic_ncols=True)
    for cc in chrom_:
        ccdepth = 0
        intra = C.matrix(balance=False, sparse=True, as_pixels=True).fetch(cc)
        ccdepth+=intra[intra['bin2_id']-intra['bin1_id']>mindis]['count'].sum()*2
        if mindis == 0:
            ccdepth+=intra[intra['bin2_id']-intra['bin1_id']==mindis]['count'].sum()
        else:
            ccdepth+=intra[intra['bin2_id']-intra['bin1_id']==mindis]['count'].sum()*2
        chrom_.desc = f'Depth of {cc}: {ccdepth}'
        totals += ccdepth
    print(f'\npolaris util depth FINISHED :)\nScaned {chrom} of {input} ({resol}bp)\nIntra reads: {totals:,}')

    
    # print('num of intra reads in your data:', totals)
    # matched_read_num = 3031042417 / genome_size * totals
    # print('num of intra reads in a human with matched sequencing coverage:', int(matched_read_num))
    # print('suggested model:', match_pretrained_models(matched_read_num))

# def match_pretrained_models(v, platform='Hi-C'):

#     if platform in ['Hi-C', 'Micro-C']:
#         arr = [
#             5000000, 10000000, 30000000, 50000000, 100000000,
#             150000000, 200000000, 250000000, 300000000, 350000000,
#             400000000, 450000000, 500000000, 550000000, 600000000,
#             650000000, 700000000, 750000000, 800000000, 850000000,
#             900000000, 1000000000, 1200000000, 1400000000, 1600000000,
#             1800000000, 2000000000
#         ]

#     diff = np.abs(v - np.r_[arr])
#     idx = np.argmin(diff)
#     if arr[idx] >= 1000000000:
#         label = '{0:.2g} billion'.format(arr[idx]/1000000000)
#     else:
#         label = '{0} million'.format(arr[idx]//1000000)

#     return label
