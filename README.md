# Matrix Balancing with Newton's Method
An implementation of a fast matrix balancing algorithm using Newton's method, which rescales a given nonnegative matrix into a [doubly stochastic matrix](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) such that each row and column sums to 1.
Please see the following paper for more details:
* Sugiyama, M., Nakahara, H., Tsuda, K.: **Tensor Balancing on Statistical Manifold**, ICML 2017 (to appear), preprint is available at *[arXiv:1702.08142](https://arxiv.org/abs/1702.08142)*.

An implementation of balancing algorithms for third-order tensors will be coming soon.


## Usage
### In your program
You can perform matrix balancing by calling the function `matrixBalancing`.
To use it, you just need to include the header file "matrixBalancing.h" in your program.
The code is written in C++11 and the [Eigen](http://eigen.tuxfamily.org) library is needed.  
The following three algorithms are implemented:
* The Newton balancing algorithm
* The Sinkhorn-Knopp algorithm
* The BNEWT algorithm proposed by [Knight and Ruiz (2013)](https://academic.oup.com/imajna/article-abstract/33/3/1029/659457/A-fast-algorithm-for-matrix-balancing?redirectedFrom=fulltext)

The main function `matrixBalancing` is defined as:
```
double matrixBalancing(MatrixXd& X, VectorXd& r, VectorXd& s, double error_tol, double rep_max, bool verbose, int32_t type)
```
* `X`: an input matrix, a balanced matrix will be returned
* `r`: a left balancer for rows will be returned
* `s`: a right balancer for columns will  be returned
* `error_tol`: error tolerance
* `rep_max`: the maximum number of iteration
* `verbose`: the verbose mode if true
* `type`: type of algorithms
  * `type = 1`: Newton balancing algorithm
  * `type = 2`: Sinkhorn-Knopp algorithm
  * `type = 3`: BNEWT algorithm
* Return value: the number of iterations

### In terminal
We provide a benchmark matrix "H20.csv" and a test code "main.cc" to try the code, which includes an input and output interface for matrix files.

For example, in the directory `src/cc`:
```
$ make
$ ./matbalance -i H20.csv
> Reading a database file "H20.csv" ... end
  Information:
  Number of (rows, cols): (20, 20)
> Removing rows and columns with all zeros ... end
  Number of (rows, cols) after removing zero rows and cols: (20, 20)
> Start Newton balancing algorithm:
  pulling zeros ... end
  Step  2, Residual: 0.717407
  Step  5, Residual: 0.0881368
  Step  8, Residual: 0.00659612
  Step 10, Residual: 0.000917773
  Step 13, Residual: 4.58902e-05
  Step 15, Residual: 6.21178e-06
  Number of iterations: 15
  Running time:         949 [microsec]
```
To compile the program, please edit paths in the "Makefile" according to the location of Eigen library in your environment.

#### Command-line arguments
* `-i <input_file>`: a path to a csv file of an input matrix (without row and column names)  
* `-o <output_matrix_file>`: an output file of the balanced matrix  
* `-a <balancer_file>`: an output file of two balancers   
* `-t <output_stat_file>`: an output file of statistics  
* `-p`: please specify it if the input file is the Matrix Market Exchange Format  
* `-e <error_tolerance>`: error tolerance is set to 1e-`<error_tolerance>` [default value: 5]  
* `-r <max_iteration>`: the maximum number of iterations is set to 1e+`<max_iteration>` [default value: 6]  
* `-v`: the verbose mode if specified  
* `-n`: the newton balancing algorithm is used  
* `-s`: the Sinkhorn-Knopp algorithm is used  
* `-b`: the BNEWT algorithm is used

## Contact
Author: Mahito Sugiyama  
Affiliation: National Institute of Informatics, Tokyo, Japan  
E-mail: mahito@nii.ac.jp
