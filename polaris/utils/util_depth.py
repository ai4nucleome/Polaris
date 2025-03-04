import click
import cooler
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

np.seterr(divide='ignore', invalid='ignore')

def process_chrom(args):
    chrom_name, input_file, resol, mindis, exclude_self = args
    try:
        C = cooler.Cooler(f"{input_file}::resolutions/{resol}")
        pixels = C.matrix(
            balance=False, sparse=True, as_pixels=True).fetch(chrom_name)
        bin_diff = pixels['bin2_id'] - pixels['bin1_id']
        min_diff = max(mindis, 1) if exclude_self else mindis
        mask = bin_diff >= min_diff
        return pixels[mask]['count'].sum()
    except Exception as e:
        print(f"Error processing {chrom_name}: {e}")
        return 0

@click.command()
@click.option('-c','--chrom', type=str, default=None, help='Comma separated chroms [all autosomes]')
@click.option('-md','--mindis', type=int, default=0, help='Min genomic distance in bins [0]')
@click.option('-r','--resol',type=int,required=True,help='Resolution (bp)')
@click.option('-i','--input', type=str,required=True,help='mcool file path')
@click.option('--exclude-self', is_flag=True, help='Exclude bin_diff=0 contacts')
def depth(input, resol, mindis, chrom, exclude_self):
    """Calculate intra-chromosomal contacts with bin distance >= mindis"""
    print(f'\n[polaris] Depth calculation START')
    
    try:
        C = cooler.Cooler(f"{input}::resolutions/{resol}")
    except ValueError:
        available_res = cooler.fileops.list_coolers(input)
        raise ValueError(f"Resolution {resol} not found. Available: {available_res}")
    
    chrom_list = chrom.split(',') if chrom else C.chromnames
    invalid_chroms = [c for c in chrom_list if c not in C.chromnames]
    if invalid_chroms:
        raise ValueError(f"Invalid chromosomes: {invalid_chroms}. Valid: {C.chromnames}")
    
    # 并行处理
    with Pool(processes=min(len(chrom_list), 4)) as pool:
        args_list = [(chrom, input, resol, mindis, exclude_self) for chrom in chrom_list]
        results = list(tqdm(pool.imap(process_chrom, args_list), total=len(chrom_list), dynamic_ncols=True))
        total_contacts = sum(results)
    
    print(f"\n[polaris] Depth calculation FINISHED")
    print(f"File: {input} (res={resol}bp)")
    print(f"Chromosomes: {chrom_list}")
    print(f"Minimum bin distance: {mindis}{', exclude self' if exclude_self else ''}")
    print(f"Total intra contacts: {total_contacts:,}")

if __name__ == '__main__':
    depth()