# My code has references to the following repositories:
# RefHiC: https://github.com/BlanchetteLab/RefHiC（Analysis code）
# Axial Attention: https://github.com/lucidrains/axial-attention (Model architecture)
# Peakachu: https://github.com/tariks/peakachu (Calculate intra reads)
# Thanks a lot for their implement.
# --------------------------------------------------------

import click
from polaris.loopScore import score
from polaris.loopLF import scorelf
from polaris.loopPool import pool
from polaris.loop import pred
from polaris.utils.util_cool2bcool import cool2bcool
from polaris.utils.util_pileup import pileup
from polaris.utils.util_depth import depth

@click.group()
def cli():
    '''
    Polaris

    A Versatile Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data
    '''
    pass

@cli.group()
def loop():
    '''Loop annotation.

    \b
    Annotate loops from chromosomal contact maps.
    '''
    pass

@cli.group()
def util():
    '''Utilities.
    
    \b
    Utilities for analysis and visualization.'''
    pass

loop.add_command(pred)
loop.add_command(score)
loop.add_command(scorelf)
loop.add_command(pool)

util.add_command(depth)
util.add_command(cool2bcool)
util.add_command(pileup)


if __name__ == '__main__':
    cli()