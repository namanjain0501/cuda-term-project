g++-11 helper.cpp
./a.out

We can get the constant values A, B, G in Algorithm for given F(m,r)
Computing the transforms of F(m, r)

Depending on m,r select m+r-2 polynomial interpolation points and hardcode the 3 values in the main function of helper.cpp. It prints the value of A, B, G as output.

These should further be hardcoded in serial-algo and parallel-algo to get the fast fourier transform.

Reference: https://github.com/andravin/wincnn/blob/a33e8def428d95c32aaf861b397de6767772ea94/wincnn.py#L46  (The author's code is given in python, we have implemented the above logic in c++)
