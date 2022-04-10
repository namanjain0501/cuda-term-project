ALGORITHM GIVEN IN THE PAPER 
1. P = N⌈H/m⌉⌈W/m⌉ is the number of image tiles. α = m + r − 1 is the input tile size.
Neighboring tiles overlap by r − 1.
2. Each tile produces mxm output
3. Concatenating all the outputs we achieve parallelism


PARALLELIZED VERSION
- Here we conside every thread performs computation for every tile
- Therefore P threads can parallely fill the output matrix o

Note:
- We can even do internal parallelization, but we avoided for sake of complexity.

