#!/usr/bin/env python3
"""Fit, predict and score the archived U2OS ASG003→LKCP001 direction."""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"frozen"))
from biomeme.demo import main
if __name__=="__main__": main(ROOT)
