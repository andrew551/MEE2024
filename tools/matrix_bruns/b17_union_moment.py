"""The union estimator on the MOMENT-mode tree (the convention-rollback L).

Executes b17_union.py with its paths redirected to matrix_bruns2017_moment/ (whose
tiers live in <tier>/stage2 rather than stage2_constant). The PS constant used for
displacement-vector conversions stays 2.0868004 -- it enters only ppm-level unit
conversions; the load-bearing scale is the host model's own stored value, which is
read from the moment-mode EA fit (frozen to the MOMENT L/R bracket).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
src = src.replace('matrix_bruns2017', 'matrix_bruns2017_moment')
src = src.replace("'stage2_constant'", "'stage2'")
exec(src)
