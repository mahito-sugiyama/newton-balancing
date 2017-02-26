# Tensor Balancing
An implementation of a fast tensor balancing algorithm by Newton's method.  
The current version supports only matrices.

## Usage
### In your program
You can perform matrix balancing by calling the function `matrixBalancing`.
To use it, you just need to include the header file "matrixBalancing.h" in your program.
The code is written in C++11 and the [Eigen](http://eigen.tuxfamily.org) library is needed.  
Three algorithms are implemented:
* The Newton balancing algorithm
* The Sinkhorn-Knopp algorithm
* The BNEWT algorithm proposed by [Knight and Ruiz](https://academic.oup.com/imajna/article-abstract/33/3/1029/659457/A-fast-algorithm-for-matrix-balancing?redirectedFrom=fulltext)

The main function `matrixBalancing` is defined as:
```
double matrixBalancing(MatrixXd& X, double error_tol, double rep_max, bool verbose, Int type);
```
* `X`: an input matrix
* `error_tol`: Error tolerance
* `rep_max`: the maximum number of iteration
* `verbose`: The verbose mode if true
* `type`: type type of algorithms
  * `type = 1`: Newton balancing algorithm
  * `type = 2`: Sinkhorn-Knopp algorithm
  * `type = 3`: BNEWT algorithm

### In terminal
To try the code, we provide a benchmark matrix "H20.csv" and a test code "main.cc", which includes input and output interface for matrix files.

For example, in the directory `src/cc`:
```
$ make
$ ./matbalance -i H20.csv
> Reading a database file "H20.csv" ... end
  Information:
  Number of (rows, cols): (20, 20)
> Removing rows and columns with all zeros ... end
  Number of (rows, cols) after removing zero rows and cols: (20, 20)
> Start Newton balancing algortihm:
  pulling zeros ... end
  Step  2, Residual: 0.717407
  Step  5, Residual: 0.0881368
  Step  8, Residual: 0.00659612
  Step 10, Residual: 0.000917773
  Step 13, Residual: 4.58902e-05
  Step 15, Residual: 6.21178e-06
  Number of iterations: 15
  Running time:         1958 [microsec]
```
To compile the program, please edit paths in the "Makefile" according to the location of Eigen library in your environment.

#### Command-line arguments

`-i <input_file>`: A path to a csv file of an input matrix (without row and column names)  
`-o <output_matrix_file>`: An output file of the balanced matrix  
`-t <output_stat_file>`: An output file of statistics
`-p`: Please spesify it if the input file is the Matrix Market Exchange Format
`-e <error_tolerance>`: Error tolerance is set to 1e-`<error_tolerance>` [default value: 5]  
`-r <max_iteration>`: The maximum number of iterations is set to 1e+`<max_iteration>` [default value: 1000000]  
`-v`: The verbose mode if specified  
`-n`: The newton balancing algorithm is used
`-s`: The Sinkhorn-Knopp algorithm is used
`-b`: The BNEWT algorithm ()is used
