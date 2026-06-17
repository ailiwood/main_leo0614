# P6A MOSEI Blocked Report

## Status

MOSEI SDK installation remains **BLOCKED**.

## Root Cause

`mmsdk` package requires setuptools < 60, but current environment has setuptools 81.0.0.
pip install with --no-deps also fails.

## Attempted Fixes

| Attempt | Result |
|---------|--------|
| pip install mmsdk | ❌ setuptools conflict |
| pip install mmsdk --no-deps | ❌ Still fails |
| Direct import | ❌ ModuleNotFoundError |

## Recommendation

1. Use MLCL/MMSA pre-extracted standard features instead of SDK
2. Or create dedicated conda env with python 3.8 + old setuptools
3. MOSEI formal training deferred until main model frozen on MOSI
