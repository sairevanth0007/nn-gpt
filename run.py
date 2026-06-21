#!/usr/bin/env python3
"""Wrapper that patches pwd.getpwuid before importing anything else.
Solves: KeyError: 'getpwuid(): uid not found: 1025' in K8s containers.
"""
import getpass, pwd

getpass.getuser = lambda: 'ab'
_orig = pwd.getpwuid
def _safe(uid):
    try: return _orig(uid)
    except KeyError: return ('ab','x',uid,1376,'','/home/ab','/bin/bash')
pwd.getpwuid = _safe

import runpy, sys
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name='__main__')