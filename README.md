# L1B Converter - converts L1A -> L1B
includes: secondary straightening

## Requirements: 
0. Bounds.csv : needed to generate line profiles
1. line profiles : profiles required to perform secondary straightening. The data typically provided for this should be strictly nighttime from one night.
    if line profiles need to be generated: 
        1. use test_line_profile.py to determine the bounds for the line and the background
        2. add those bounds to bounds.csv
        3. run generate_line_profile.py to create line profiles for all the windows for which the bounds are provided.

## to install secondary_straightening:

```
python -m build
pip install .
