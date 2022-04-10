#include <iostream>

#include <bits/stdc++.h>

#include<vector>
#include<algorithm>
#include<string>
#include<iostream>
#include <typeinfo>

using namespace std;

#define FractionsInG 0
#define FractionsInA 1
#define FractionsInB 2
#define FractionsInF 3

// Caluclation of A
void def_At (float *a, int m, int n, float **At);
void def_A (float *a, int m, int n, float **A);

// Caluclation of B
void def_T (float *a, int n, float **T);
long coeff_a_poly_b(int a, int b, int n);
string repeat(string s, int n);
void def_F (float *a, int n, float **F);
void def_Fdiag (float *a, int n, float **Fdiag);
void def_FdiagPlus1 (float *a, int n, float **FdiagPlus1);
void def_L (float *a, int n, float **L);
void def_Bt (float *a, int n, float **Bt);
void def_B (float *a, int n, float **B);


// Matrix Operations
float **transpose (float **A, int m, int n);
float **matrix_mult (float **A, int m, int n, float **B, int r);
void getCofactor(float **A, float **temp, int N, int p, int q, int n);
void adjoint(float **A, float **adj, int N);
bool inverse(float **A, float **inverse, int N);

// Debug Functions
void printMatrix(float **A, int m, int n);
float** newMatrix(int m, int n) ;

void findMatrices (float *a, int n, int r, float **AT, float **G, float **BT, float **f, int fractionsIn = FractionsInG);

// Output Matrix At : m x n
void def_At (float *a, int m, int n, float **At) {
    for (int i = 0; i < m; i++) {
        for (int j = 0;  j< n; j++) {
            At[i][j] = pow(a[i], j);
        }
    }
}

// Output Matrix A : m x n
void def_A (float *a, int m, int n, float **A) {
    def_At (a, m-1, n, A);
    for (int j = 0; j < n; j++) {
        A[m-1][j] = (j==n-1)?1:0;
    }
}

void printMatrix(float **A, int m, int n) {
	int i,  j;
	for(i=0; i<m; i++) {
		for(j=0; j<n; j++) {
			cout<<A[i][j]<<" ";
		}
		cout<<"\n";
	}
}

// Transposes matrix A of dimensions (m x n)
// Returns transpose of dimensions (n x m)
float **transpose (float **A, int m, int n) {
    float **T = (float**) malloc (n*sizeof(float*));
    for (int i = 0; i < n; i++) {
        T[i] = (float*) malloc (m*sizeof(float));
        for (int j = 0; j < m; j++) {
            T[i][j] = A[j][i];
        }
    }
    return T;
}

float** newMatrix(int m, int n) {
	float** new_matrix;
	new_matrix = (float**)malloc(m*sizeof(float*));
	for(int i =0; i<m; i++) {
		new_matrix[i] = (float*) malloc(n*sizeof(float));
	}
	return new_matrix;
}

// Output Matrix T : n x (n+1)
void def_T (float *a, int n, float **T) {
	
	for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
					T[i][j] = (i==j)?1:0;
			}
	}
	for (int i = 0; i < n; i++) {
			T[i][n] = pow(-a[i], n);
	}	
}


// Coefficient of x^a in Polynomial whose roots are 0 to n-1 except b
long coeff_a_poly_b(float *A, int a, int b, int n) {
    // initalize roots array
    vector<int> roots;
		string s, s1, s2;
    long value, sum;
		int sign = 1;

    for(int i = 0; i<n; i++)
      if(i!=b)
        roots.push_back(A[i]);

    // string with a 1s and rest 0s
		s1 = "a";
	  s2 = "b";
    s1 = repeat(s1, a);
    s2 = repeat(s2,n-1-a );

    sum = 0;

	 	s = string(s1) + string(s2);
		s = string(s);
	
    // Magnitude of coefficient
    do {
			value = 1;
			for(int i=0; i<(n-1); i++)
				if(s[i]=='b')
					value = value * roots[i];
			sum += value;
			// cout<<s1<<" ";
            
    } while (next_permutation(s.begin(), s.end()));

    // Sign of value (-1)^a
    if(a%2)
      sign = -1;

    return sum*sign;
}

// Function which return string by concatenating it.
string repeat(string s, int n) {
	if(n==0)
		return "";
	
	string s1 = s;

	for (int i=1; i<n;i++)
		s = s + string(s1);

	return s;
}

// Output Matrix F : n x 1
void def_F (float *a, int n, float **F) {
    for (int i = 0; i < n; i++) {
        F[i][0] = 1;
        for (int j = 0; j < n; j++) {
            F[i][0] *= (j!=i)?(a[i]-a[j]):1;
        }
    }
}

// Output Matrix Fdiag : n x n
void def_Fdiag (float *a, int n, float **Fdiag) {
    float **f = (float**) malloc (n*sizeof(float*));
    for (int i = 0; i < n; i++) {
        f[i] = (float*) malloc (n*sizeof(float));
    }

    def_F(a, n, f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Fdiag[i][j] = (i==j)?f[i][0]:0;
        }
    }
}

// Output Matrix FdiagPlus1 : n x n
void def_FdiagPlus1 (float *a, int n, float **FdiagPlus1) {
    def_Fdiag (a, n-1, FdiagPlus1);

    for (int i = 0; i < n; i++) {
        FdiagPlus1[i][n-1] = 0;
    }

    for (int j = 0; j < n; j++) {
        FdiagPlus1[n-1][j] = (j==n-1)?1:0;
    }
}

// Output Matrix L : n x n
void def_L (float *a, int n, float **L) {
	float **F = (float**) malloc (n*sizeof(float*));
	for (int i = 0; i < n; i++) {
			F[i] = (float*) malloc (n*sizeof(float));
	}
	def_F (a, n, F);
	// cout<<"____________F___________\n";
	// printMatrix(F, n, 1);
	
	float **L1, **L2;
	int temp;
	L1 = newMatrix(n, n);

	// cout<<"\n";
	for(int i = 0; i<n; i++) {
		for(int j = 0; j<n; j++) {
			temp = coeff_a_poly_b(a, j, i, n);
			if(temp == 0) {
				L1[i][j] = 0;
			} else 
					L1[i][j] = (float) temp / F[i][0];
			// cout<<temp<<" ";
		}
		// cout<<"\n";
	}

	L2 = transpose(L1, n, n);
	for(int i =0; i<n; i++) {
		for(int j =0 ;j<n; j++) {
			L[i][j] = L2[i][j];
		}
	}
		
}

// Output Matrix Bt : n x (n+1)
void def_Bt (float *a, int n, float **Bt) {
    float **L = (float**) malloc (n*sizeof(float*));
    for (int i = 0; i < n; i++) {
        L[i] = (float*) malloc (n*sizeof(float));
    }
    def_L (a, n, L);
	// cout<<"___________L__________\n";
		// printMatrix(L, n , n);
	
    float **T = (float**) malloc (n*sizeof(float*));
    for (int i = 0; i < n; i++) {
          T[i] = (float*) malloc ((n+1)*sizeof(float));
    }
    def_T (a, n, T);
	  // cout<<"___________T__________\n";
		// printMatrix(T, n , n+1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n+1; j++) {
            Bt[i][j] = 0;
            for (int k = 0; k < n; k++) {
                Bt[i][j] += L[i][k]*T[k][j];
            }
        }
    }
}

// Output Matrix B : n x n
void def_B (float *a, int n, float **B) {
    def_Bt (a, n-1, B);
		// cout<<"___________Bt__________\n";
		// printMatrix(B, n-1 , n);
    for (int j = 0; j < n; j++) {
        B[n-1][j] = (j==n-1)?1:0;
    }
}

// Multiplies matrix A of dimensions (m x n) with matrix B of dimensions (n x r)
// Returns matrix of dimensions (m x r)
float **matrix_mult (float **A, int m, int n, float **B, int r) {
    float **M = (float**) malloc (m*sizeof(float*));
    for (int i = 0; i < m; i++) {
        M[i] = (float*) malloc (r*sizeof(float));
        for (int j = 0; j < r; j++) {
            M[i][j] = 0;
            for (int k = 0; k < n; k++) {
                M[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return M;
}

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(float **A, float **temp, int N, int p, int q, int n) {
    int i = 0, j = 0;
 
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
// Recursive function for finding determinant of matrix A of dimension N x N
// n is current dimension of A
float determinant(float **A, int N, int n) {
    float D = 0; // Initialize result
 
    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];
 
    float **temp; // To store cofactors
    temp = (float**) malloc (N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        temp[i] = (float*) malloc (N*sizeof(float));
    }
 
    int sign = 1;  // To store sign multiplier
 
     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, N, 0, f, n);
        D += sign * A[0][f] * determinant(temp, N, n - 1);
 
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}
 
// Function to get adjoint of matrix A of size N x N in adj matrix.
void adjoint(float **A, float **adj, int N) {
    if (N == 1) {
        adj[0][0] = 1;
        return;
    }
 
    // temp is used to store cofactors of A[][]
    int sign = 1;
    float **temp = (float**) malloc (N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        temp[i] = (float*) malloc (N*sizeof(float));
    }
 
    for (int i = 0; i < N; i++)
    {
        for (int j  =0; j < N; j++)
        {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, N, i, j, N);
 
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;
 
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign)*(determinant(temp, N, N-1));
        }
    }
}


// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(float **A, float **inverse, int N) {
    // Find determinant of A[][]
    float det = determinant(A, N, N);
    if (det == 0) {
        return false;
    }
 
    // Find adjoint
    float **adj = (float**) malloc (N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        adj[i] = (float*)  malloc (N*sizeof(float));
    }
    adjoint(A, adj, N);
 
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inverse[i][j] = adj[i][j]/det;
        }
    }
 
    return true;
}


// Populates AT, G, BT and f
void findMatrices (float *a, int n, int r, float **AT, float **G, float **BT, float **f, int fractionsIn) {
    int alpha = n+r-1;
    def_FdiagPlus1 (a, alpha, f);

	
	

    if (f[0][0] < 0) {
        for (int j = 0; j < alpha; j++) {
            f[0][j] *= -1;
        }
    }

		float **A, **X, **XT, **fT, **Z, **ZT;
	  A = newMatrix(alpha, n);

		printMatrix(f, alpha, alpha);
	cout<<"---------";
		fT = newMatrix(alpha, alpha);
	inverse(f, fT, alpha);
	printMatrix(fT, alpha, alpha);
	cout<<"----------";
		Z = newMatrix(alpha, alpha);


    if (fractionsIn == FractionsInG) {
        def_A (a, alpha, n, A);
        AT = transpose(A, alpha, n);

        X = newMatrix(alpha, r);
        def_A (a, alpha, r, X);
        XT = transpose (X, alpha, r);
        
        float **Y = matrix_mult (XT, r, alpha, fT, alpha);
        G = transpose (Y, r, alpha);

        def_B (a, alpha, Z);
        ZT = transpose (Z, alpha, alpha);
        BT = matrix_mult (f, alpha, alpha, ZT, alpha);

    } else if (fractionsIn == FractionsInA) {
        float **X = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            X[i] = (float*) malloc (alpha*sizeof(float));
        }
        def_B (a, alpha, X);
        float **XT = transpose (X, alpha, alpha);
        BT = matrix_mult (f, alpha, alpha, XT, alpha);

        G = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            G[i] = (float*) malloc (r*sizeof(float));
        }
        def_A (a, alpha, r, G);

        float **Y = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            Y[i] = (float*) malloc (n*sizeof(float));
        }
        def_A (a, alpha, n, Y);
        float **YT = transpose (Y, alpha, n);
        float **fT = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            fT[i] = (float*) malloc (alpha*sizeof(float));
        }
        inverse(f, fT, alpha);
        AT = matrix_mult (YT, n, alpha, fT, alpha);

    } 
		else if (fractionsIn = FractionsInB) {
        def_A (a, alpha, n, A);
        AT = transpose(A, alpha, n);

        G = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            G[i] = (float*) malloc (r*sizeof(float));
        }
        def_A (a, alpha, r, G);

        float **B = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            B[i] = (float*) malloc (alpha*sizeof(float));
        }
        def_B (a, alpha, B);
        BT = transpose (B, alpha, alpha);

    } 
		else {
        def_A (a, alpha, n, A);
        AT = transpose(A, alpha, n);

        G = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            G[i] = (float*) malloc (r*sizeof(float));
        }
        def_A (a, alpha, r, G);

        float **X = (float**) malloc (alpha*sizeof(float*));
        for (int i = 0; i < alpha; i++) {
            X[i] = (float*) malloc (alpha*sizeof(float));
        }
        def_B (a, alpha, X);
        float **XT = transpose (X, alpha, alpha);
        BT = matrix_mult (f, alpha, alpha, XT, alpha);
    }

	cout<<"----------------------\n";
	printMatrix(AT, n, alpha);
	fflush(stdout);
	cout<<"----------------------\n";
	printMatrix(BT, n, n);
	
	cout<<"----------------------\n";
	printMatrix(G, n, n);
	
}


int main() {
	// For F(m,r) you must select m+r-2 polynomial interpolation points.
	// FOR NOW HARDCODING

	float *a;
	int m, r, i, j, n;
	m = 2, r =3;
	n = m+r-1;
	
	a = (float*)malloc(n*sizeof(float));
	a[0] = 0;
	a[1] = 1;
	a[2] = -1;

	float **A, **AT; 
	float **B, **BT;
	float **G, **f;

	AT = newMatrix(m, n);
	BT = newMatrix(n, n);
	G = newMatrix(m, n);
	f = newMatrix(n, n);

	findMatrices(a, m, r, AT, G, BT, f, FractionsInG);


	// A = newMatrix(n, m); // nxm
	// B = newMatrix(n, n); // nxn

	// cout<<"___________A__________\n";
	// def_A(a, n, m, A);
	// printMatrix(A,n, m);
	// cout<<"___________AT__________\n";
	// AT = transpose(A, n, m);
	// printMatrix(AT, m , n);
	// cout<<"___________B__________\n";
	// def_B(a, n, B);
	// printMatrix(B,n, n);
	// cout<<"___________BT__________\n";
	// BT = transpose(B, n, n);
	// printMatrix(BT, n , n);
	
	return 0;
}