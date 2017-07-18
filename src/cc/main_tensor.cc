/*
    An input and output interface for matrix balancing
    Copyright (C) 2017  Mahito Sugiyama

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "tensorBalancing.h"
#include "MarketIO.h"
#include <unistd.h>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main(int argc, char *argv[]) {
  bool verbose = false;
  bool flag_in = false;
  bool flag_out = false;
  bool flag_balancer = false;
  bool flag_stat = false;
  bool do_newton = true;
  bool do_sinkhorn = false;
  bool do_bnewt = false;
  double rep_max = 1e+06;
  double error_tol = 1e-05;
  char *input_file = NULL;
  char *output_file = NULL;
  char *balancer_file = NULL;
  char *stat_file = NULL;
  Int num_mat = 1;

  // get arguments
  char opt;
  while ((opt = getopt(argc, argv, "i:o:t:e:r:vnsbd:")) != -1) {
    switch (opt) {
    case 'i': input_file = optarg; flag_in = true; break;
    case 'o': output_file = optarg; flag_out = true; break;
    case 't': stat_file = optarg; flag_stat = true; break;
    case 'e': error_tol = pow(10, -1 * atof(optarg)); break;
    case 'r': rep_max = pow(10, atof(optarg)); break;
    case 'v': verbose = true; break;
    case 'n': do_newton = true; do_sinkhorn = false; do_bnewt = false; break;
    case 's': do_newton = false; do_sinkhorn = true; do_bnewt = false; break;
    case 'b': do_newton = false; do_sinkhorn = false; do_bnewt = true; break;
    case 'd': num_mat = atoi(optarg); break;
    }
  }

  if (!flag_in) {
    cerr << "> ERROR: Input file (-i [input_file]) is missing!" << endl;
    exit(1);
  }
  ofstream sfs;
  if (flag_stat) {
    sfs.open(stat_file);
  }

  cout << "> Reading a database file \"" << input_file << "\" ... " << flush;
  vector<vector<vector<double>>> X;
  ifstream ifs(input_file);
  if (!ifs) {
    cerr << endl << "  ERROR: The file \"" << input_file << "\" does not exist!!" << endl;
    exit(1);
  }
  readTensorFromCSV(X, num_mat, ifs);
  ifs.close();

  cout << "end" << endl << flush;
  cout << "  Information:" << endl << flush;
  cout << "  Size: (" << X.size() << ", " << X.front().size() << ", " << X.front().front().size() << ")" << endl << flush;

  // check whether an input matrix is nonnegative
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x < 0.0) {
	  cout << "Negative value exists!" << endl;
	  exit(1);
	}
      }
    }
  }

  if (flag_stat) {
    sfs << "Number_of_dim1:\t" << X.front().front().size() << endl;
    sfs << "Number_of_dim2:\t" << X.front().size() << endl;
    sfs << "Number_of_dim3:\t" << X.size() << endl;
  }

  Int type = 1;
  if (do_newton) {
    cout << "> Start Newton balancing algortihm:" << endl << flush;
    type = 1;
  } else if (do_sinkhorn) {
    cout << "> Start the Sinkhorn-Knopp balancing algorithm:" << endl << flush;
    type = 2;
  }

  // run tensor balancing
  auto ts = system_clock::now();
  double step = TensorBalancing(X, error_tol, rep_max, verbose, type);
  auto te = system_clock::now();
  auto dur = te - ts;
  cout << "  Number of iterations: " << step << endl;
  cout << "  Running time:         " << duration_cast<microseconds>(dur).count() << " [microsec]" << endl;
  if (flag_out) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
    ofstream ofs(output_file);
    ofs << X;
    ofs.close();
  }
  if (flag_stat) {
    sfs << "Number_of_iterations:\t" << step << endl;
    sfs << "Running_time_(microsec):\t" << duration_cast<microseconds>(dur).count() << endl;
    sfs.close();
  }
}
