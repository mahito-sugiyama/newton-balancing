/*
    A Newton balancing algorithm for tensor balancing
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

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <functional>
#include <utility>
#include <iomanip>
#include <tuple>
#include <time.h>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define Int int32_t
#define Tensor vector<vector<vector<double>>>
#define Poset vector<vector<vector<node>>>
#define PosetIndex vector<vector<pair<Int, Int>>>
#define D 3

using namespace std;
using namespace Eigen;
using namespace std::chrono;

double EPSILON = 1e-300;

// node structure
class node {
public:
  bool is_lower;
  bool is_upper;
  Int id, id_org;
  Int check;
  Int supp;
  pair<Int, Int> id_mat;
  double p, p_init, p_prev;
  double theta, theta_init, theta_prev, theta_sum, theta_sum_prev;
  double eta, eta_init, eta_prev;
  double score, pvalue;
  vector<reference_wrapper<node>> from;
  vector<reference_wrapper<node>> to;
};

// output a poset "s"
ostream &operator<<(ostream& out, const vector<node>& s) {
  Int width = 8;
  out << setw(4) << right << "id" << setw(width) << right << "prob" << setw(width) << right << "theta" << setw(width) << right << "eta" << endl;
  for (auto&& x : s) {
    out << setw(4) << right << setprecision(4) << fixed << x.id << setw(width) << right << x.p << setw(width) << right << x.theta << setw(width) << right << x.eta << endl;
  }
  return out;
}
// output a tensor
ostream &operator<<(ostream& out, const Tensor& X) {
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (Int i = 0; i < (Int)vec.size() - 1; ++i) {
	out << vec[i] << ", ";
      }
      out << vec.back() << endl;
    }
    out << endl;
  }
  return out;
}

// for "reverse" in range-based loop
template<class Cont> class const_reverse_wrapper {
  const Cont& container;
public:
  const_reverse_wrapper(const Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() const { return container.rbegin(); }
  decltype(container.rend()) end() const { return container.rend(); }
};
template<class Cont> class reverse_wrapper {
  Cont& container;
public:
  reverse_wrapper(Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() { return container.rbegin(); }
  decltype(container.rend()) end() { return container.rend(); }
};
template<class Cont> const_reverse_wrapper<Cont> reverse(const Cont& cont) {
  return const_reverse_wrapper<Cont>(cont);
}
template<class Cont> reverse_wrapper<Cont> reverse(Cont& cont) {
  return reverse_wrapper<Cont>(cont);
}

// read a database file
void readTensorFromCSV(Tensor& X, Int num_mat, ifstream& ifs) {
  vector<vector<double>> data;
  string line;
  while (getline(ifs, line)) {
    stringstream lineStream(line);
    string cell;
    vector<double> tmp;
    while (getline(lineStream, cell, ',')) {
      tmp.push_back(stod(cell));
    }
    data.push_back(tmp);
  }

  if (data.size() % num_mat != 0) {
    cerr << endl << "The size specification of the input tensor (= " << num_mat << ") is invalid!" << endl;
    exit(1);
  }

  X = Tensor(num_mat, vector<vector<double>>(data.size() / num_mat, vector<double>(data.front().size(), 0)));

  for (Int k = 0; k < num_mat; ++k) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int i = 0; i < X.front().front().size(); ++i) {
	X[k][j][i] = data[j + k * X.front().size()][i];
      }
    }
  }
}


// ================================================ //
// ========== Newton balancing algorithm ========== //
// ================================================ //
// preprocess X by a one step Sinkhorn balancing
void preprocess(Tensor& X) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  double X_sum = 0.0;
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      X_sum += accumulate(vec.begin(), vec.end(), 0.0);
    }
  }
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x /= X_sum;
      }
    }
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      double sum = 0.0;
      for (Int k = 0; k < n3; ++k) {
	sum += X[i][j][k];
      }
      for (Int k = 0; k < n3; ++k) {
	X[i][j][k] /= (sum * n1 * n2);
      }
    }
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int k = 0; k < n3; ++k) {
      double sum = 0.0;
      for (Int j = 0; j < n2; ++j) {
	sum += X[i][j][k];
      }
      for (Int j = 0; j < n2; ++j) {
	X[i][j][k] /= (sum * n1 * n3);
      }
    }
  }
  for (Int j = 0; j < n2; ++j) {
    for (Int k = 0; k < n3; ++k) {
      double sum = 0.0;
      for (Int i = 0; i < n1; ++i) {
	sum += X[i][j][k];
      }
      for (Int i = 0; i < n1; ++i) {
	X[i][j][k] /= (sum * n2 * n3);
      }
    }
  }
}
// pull away zero entries to right-upper and left-lower corners
bool checkPullCondition(Tensor& X, vector<vector<Int>>& idx_sorted) {
  for (Int d1 = 0; d1 < D; ++d1) {
    Int d2 = d1 + 1; if (d2 > 2) d2 -= D;
    Int d3 = d1 + 2; if (d3 > 2) d3 -= D;
    for (auto&& i : idx_sorted[d1]) {
      vector<Int> first_nonzero;
      for (auto&& j : idx_sorted[d2]) {
	for (Int k = 0; k < (Int)idx_sorted[d3].size(); ++k) {
	  Int i1, i2, i3;
	  if (d1 == 0) {
	    i1 = i; i2 = j; i3 = idx_sorted[d3][k];
	  } else if (d1 == 1) {
	    i1 = idx_sorted[d3][k]; i2 = i; i3 = j;
	  } else if (d1 == 2) {
	    i1 = j; i2 = idx_sorted[d3][k]; i3 = i;
	  }
	  if (X[i1][i2][i3] > EPSILON) {
	    first_nonzero.push_back(k);
	    break;
	  }
	}
      }
      for (Int ii = 0; ii < (Int)first_nonzero.size() - 1; ++ii) {
	if (first_nonzero[ii] > first_nonzero[ii + 1]) {
	  return false;
	}
      }
      first_nonzero.clear();
      for (auto&& k : idx_sorted[d3]) {
	vector<Int> first_nonzero;
	for (Int j = 0; j < (Int)idx_sorted[d2].size(); ++j) {
	  Int i1, i2, i3;
	  if (d1 == 0) {
	    i1 = i; i2 = idx_sorted[d2][j]; i3 = k;
	  } else if (d1 == 1) {
	    i1 = k; i2 = i; i3 = idx_sorted[d2][j];
	  } else if (d1 == 2) {
	    i1 = idx_sorted[d2][j]; i2 = k; i3 = i;
	  }
	  if (X[i1][i2][i3] > EPSILON) {
	    first_nonzero.push_back(j);
	    break;
	  }
	}
      }
      for (Int ii = 0; ii < (Int)first_nonzero.size() - 1; ++ii) {
	if (first_nonzero[ii] > first_nonzero[ii + 1]) {
	  return false;
	}
      }
      first_nonzero.clear();
    }
  }
  return true;
}
void pullZerosEach(Tensor& X, vector<vector<Int>>& idx_sorted) {
  vector<Int> nonzero_vec;
  vector<Int> zero_vec;

  for (Int d1 = 0; d1 < D; ++d1) {
    Int d2 = d1 + 1; if (d2 > 2) d2 -= D;
    Int d3 = d1 + 2; if (d3 > 2) d3 -= D;
    for (auto&& i : reverse(idx_sorted[d1])) {
      for (auto&& j : reverse(idx_sorted[d2])) {
	for (auto&& k : idx_sorted[d3]) {
	  Int i1, i2, i3;
	  if (d1 == 0) {
	    i1 = i; i2 = j; i3 = k;
	  } else if (d1 == 1) {
	    i1 = k; i2 = i; i3 = j;
	  } else if (d1 == 2) {
	    i1 = j; i2 = k; i3 = i;
	  }
	  if (X[i1][i2][i3] > EPSILON) {
	    nonzero_vec.push_back(k);
	  } else {
	    zero_vec.push_back(k);
	  }
	}
	idx_sorted[d3].clear();
	for (auto&& x : nonzero_vec) {
	  idx_sorted[d3].push_back(x);
	}
	for (auto&& x : zero_vec) {
	  idx_sorted[d3].push_back(x);
	}
	nonzero_vec.clear();
	zero_vec.clear();
      }
    }
  }
}
void pullZeros(Tensor& X, vector<vector<Int>>& idx_sorted) {
  while (!checkPullCondition(X, idx_sorted)) {
    pullZerosEach(X, idx_sorted);
  }
}
// make a node index
void makePosetIndex(Tensor& X, PosetIndex& idx_tp) {
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};

  // traverse a matrix in the topological order
  for (Int ii = 0; ii < D; ++ii) {
    Int i1 = ii;
    Int i2 = ii + 1; if (i2 > 2) i2 -= D;
    Int i3 = ii + 2; if (i3 > 2) i3 -= D;
    for (Int i = 0; i < n[i2] + n[i3] - 1; ++i) {
      Int j = i + 1 - n[i2];
      if (j < 0) j = 0;
      for (; j <= min(i, n[i3] - 1); j++) {
	idx_tp[i1].push_back(make_pair(i - j, j));
      }
    }
  }
}
// make a node matrix from eigen matrix
void makePosetTensor(Tensor& X, Poset& S ,vector<vector<Int>>& idx_sorted) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  // initialization
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	// S[i][j][k].p = X[i][j][k];
	S[i][j][k].p = X[idx_sorted[0][i]][idx_sorted[1][j]][idx_sorted[2][k]];
	S[i][j][k].theta = 0; S[i][j][k].theta_sum = 0; S[i][j][k].theta_sum_prev = 0;
	S[i][j][k].eta = 0;
      }
    }
  }
}
// make a beta the submanifold
void makeBeta(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  vector<Int> n{(Int)S.size(), (Int)S.front().size(), (Int)S.front().front().size()};
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x.check = false;
      }
    }
  }

  beta.push_back(make_pair(make_tuple(0, 0, 0), 1.0));
  S[0][0][0].check = true;

  for (int id = 0; id < D; ++id) {
    Int id2 = id + 1; if (id2 > 2) id2 -= D;
    Int id3 = id + 2; if (id3 > 2) id3 -= D;
    for (Int j = 0; j < n[id2]; ++j) {
      for (Int k = 0; k < n[id3]; ++k) {
	if (!(j == 0 && k == 0)) {
	  for (Int i = 0; i < n[id]; ++i) {
	    Int s1, s2, s3;
	    if (id == 0) {
	      s1 = i; s2 = j; s3 = k;
	    } else if (id == 1) {
	      s1 = k; s2 = i; s3 = j;
	    } else if (id == 2) {
	      s1 = j; s2 = k; s3 = i;
	    }
	    if (S[s1][s2][s3].p > EPSILON) {
	      if (!S[s1][s2][s3].check) {
		beta.push_back(make_pair(make_tuple(s1, s2, s3), ((double)(n[id2] - j) / (double)n[id2]) * ((double)(n[id3] - k) / (double)n[id3])));
		S[s1][s2][s3].check = true;
	      }
	      break;
	    }
	  }
	}
      }
    }
  }
}
// compute theta for beta
void computeTheta(Poset& S, PosetIndex& idx_tp) {
  S[0][0][0].theta = log(S[0][0][0].p);
  S[0][0][0].theta_sum = log(S[0][0][0].p);

  for (Int i = 0; i < S.size(); ++i) {
    for (Int j = 0; j < idx_tp[0].size(); ++j) {
      // cout << "(" << i << ", " << idx_tp[0][j].first << ", " << idx_tp[0][j].second << ")" << endl;
      if (S[i][idx_tp[0][j].first][idx_tp[0][j].second].p > EPSILON) {
	double theta_sum = 0.0;
	for (Int i2 = i; i2 >= 0; --i2) {
	  for (Int j2 = j; j2 >= 0; --j2) {
	    if (i - i2 >= 0 && idx_tp[0][j].first - idx_tp[0][j2].first >= 0 && idx_tp[0][j].second - idx_tp[0][j2].second >= 0
		&& S[i2][idx_tp[0][j2].first][idx_tp[0][j2].second].p > EPSILON) {
	      theta_sum += S[i2][idx_tp[0][j2].first][idx_tp[0][j2].second].theta;
	    }
	  }
	}
	theta_sum -= S[i][idx_tp[0][j].first][idx_tp[0][j].second].theta;
	S[i][idx_tp[0][j].first][idx_tp[0][j].second].theta = log(S[i][idx_tp[0][j].first][idx_tp[0][j].second].p) - theta_sum;
	S[i][idx_tp[0][j].first][idx_tp[0][j].second].theta_sum = S[i][idx_tp[0][j].first][idx_tp[0][j].second].theta + theta_sum;
      }
    }
  }
}
// compute eta for all entries
void computeEta(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	S[i][j][k].eta = 0.0;
	for (Int i2 = i; i2 < n1; ++i2) {
	  for (Int j2 = j; j2 < n2; ++j2) {
	    for (Int k2 = k; k2 < n3; ++k2) {
	      S[i][j][k].eta += S[i2][j2][k2].p;
	    }
	  }
	}
      }
    }
  }
}
// e-projection
void computeP(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	if (S[i][j][k].p > EPSILON) {
	  double theta_sum = 0.0;
	  double theta_sum_prev = 0.0;
	  for (Int i2 = i; i2 >= 0; --i2) {
	    for (Int j2 = j; j2 >= 0; --j2) {
	      for (Int k2 = k; k2 >= 0; --k2) {
		if (S[i2][j2][k2].p > EPSILON) {
		  theta_sum += S[i2][j2][k2].theta;
		  theta_sum_prev += S[i2][j2][k2].theta_prev;
		}
	      }
	    }
	  }
	  S[i][j][k].theta_sum = theta_sum;
	  S[i][j][k].theta_sum_prev = theta_sum_prev;
	  S[i][j][k].p = exp(theta_sum);
	}
      }
    }
  }
}
void renormalize(Poset& S) {
  // total sum
  double p_sum = 0.0;
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) p_sum += x.p;
      }
    }
  }
  // store the previous theta
  S[0][0][0].theta_prev = S[0][0][0].theta;
  // update theta(\bot)
  S[0][0][0].theta = S[0][0][0].theta_prev - log(p_sum);
  // update p
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) x.p *= exp(S[0][0][0].theta - S[0][0][0].theta_prev);
      }
    }
  }
}
void eProject(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  Int S_size = 0;
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) S_size++;
      }
    }
  }
  VectorXd theta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec = VectorXd::Zero(beta.size());
  for (Int i = 0; i < beta.size(); i++) {
    theta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    eta_vec[i]   = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second;
  }
  MatrixXd J(beta.size(), beta.size()); // Jacobian matrix
  for (Int i1 = 0; i1 < beta.size(); i1++) {
    for (Int i2 = 0; i2 < beta.size(); i2++) {
      Int i1_i = get<0>(beta[i1].first);
      Int i1_j = get<1>(beta[i1].first);
      Int i1_k = get<2>(beta[i1].first);
      Int i2_i = get<0>(beta[i2].first);
      Int i2_j = get<1>(beta[i2].first);
      Int i2_k = get<2>(beta[i2].first);
      J(i1, i2) = S[max(i1_i, i2_i)][max(i1_j, i2_j)][max(i1_k, i2_k)].eta;
      J(i1, i2) -= S[i1_i][i1_j][i1_k].eta * S[i2_i][i2_j][i2_k].eta * (double)S_size;
    }
  }

  theta_vec += (-1 * J).colPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivLu().solve(eta_vec);

  // store theta
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta_prev = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta = theta_vec[i];
  }
  // update p
  computeP(S);
  renormalize(S);
  computeEta(S);
}
// compute the residual
double computeResidual(Tensor& X) {
  double res = 0.0;
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      double sum = 0.0;
      for (Int k = 0; k < n3; ++k) {
	sum += X[i][j][k];
      }
      res += pow(sum - (1.0 / (double)(n1 * n2)), 2.0);
    }
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int k = 0; k < n3; ++k) {
      double sum = 0.0;
      for (Int j = 0; j < n2; ++j) {
	sum += X[i][j][k];
      }
      res += pow(sum - (1.0 / (double)(n1 * n3)), 2.0);
    }
  }
  for (Int j = 0; j < n2; ++j) {
    for (Int k = 0; k < n3; ++k) {
      double sum = 0.0;
      for (Int i = 0; i < n1; ++i) {
	sum += X[i][j][k];
      }
      res += pow(sum - (1.0 / (double)(n2 * n3)), 2.0);
    }
  }
  return sqrt(res);
}
void recoverZeros(Tensor& X, vector<vector<Int>>& idx_sorted) {
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};
  vector<vector<Int>> idx_org(D);
  for (Int i = 0; i < D; ++i) {
    idx_org[i].resize(n[i]);
    iota(idx_org[i].begin(), idx_org[i].end(), 0);
    sort(idx_org[i].begin(), idx_org[i].end(), [&idx_sorted, i](Int j1, Int j2) {return idx_sorted[i][j1] < idx_sorted[i][j2];});
  }

  Tensor Xnew = Tensor(n[0], vector<vector<double>>(n[1], vector<double>(n[2])));

  for (Int i = 0; i < n[0]; ++i) {
    for (Int j = 0; j < n[1]; ++j) {
      for (Int k = 0; k < n[2]; ++k) {
	Xnew[i][j][k] = X[idx_org[0][i]][idx_org[1][j]][idx_org[2][k]];
      }
    }
  }
  for (Int i = 0; i < n[0]; ++i) {
    for (Int j = 0; j < n[1]; ++j) {
      for (Int k = 0; k < n[2]; ++k) {
	X[i][j][k] = Xnew[i][j][k];
      }
    }
  }
}
// the main function for newton balancing algorithm
double NewtonBalancing(Tensor& X, double error_tol, double rep_max, bool verbose) {
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();
  clock_t ts, te;
  // preprocess
  preprocess(X);
  // make a node matrix
  Poset S = Poset(n1, vector<vector<node>>(n2, vector<node>(n3)));
  PosetIndex idx_tp(D);
  makePosetIndex(X, idx_tp);
  cout << "  pulling zeros ... " << flush;
  vector<vector<Int>> idx_sorted(D);
  for (Int i = 0; i < D; ++i) {
    idx_sorted[i].resize(n[i]);
    iota(idx_sorted[i].begin(), idx_sorted[i].end(), 0);
  }
  pullZeros(X, idx_sorted);
  cout << "end" << endl;

  makePosetTensor(X, S, idx_sorted);

  vector<pair<tuple<Int, Int, Int>, double>> beta;
  makeBeta(S, beta);
  cout << "  Number of constraint: " << beta.size() << endl;
  computeTheta(S, idx_tp);
  computeEta(S);
  // run Newton's method
  if (verbose) cout << "----- Start Newton's method -----" << endl;
  double res = 0.0;
  double step = 1.0;
  Int exponent = 0;
  auto t_start = system_clock::now();
  while (step <= rep_max) {
    // perform e-projection
    eProject(S, beta);
    // put results to X
    for (Int i = 0; i < n1; ++i) {
      for (Int j = 0; j < n2; ++j) {
	for (Int k = 0; k < n3; ++k) {
	  X[i][j][k] = S[i][j][k].p;
	}
      }
    }
    double res_prev = res;
    res = computeResidual(X);
    if (res_prev >= EPSILON && res > res_prev * 100) {
      cout << "Current residual = " << res << endl;
      cout << "Terminate with failing to convergence." << endl;
      return step;
    }

    // output the residual
    if (verbose) {
      cout << "Step\t" << step << "\t" << "Residual\t" << res << endl << flush;
    } else {
      if (res < pow(10, -1.0 * (double)exponent)) {
	cout << "  Step "; if (step < 10.0) cout << " ";
	cout << step << ", Residual: " << res << endl;
	exponent++;
      }
    }
    if (res < error_tol) break;
    step += 1.0;
  }
  if (verbose) cout << "----- End Newton's method -----" << endl;
  // swap again to recover the original row and column orders
  recoverZeros(X, idx_sorted);
  sort(n.begin(), n.end(), greater<int>());
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x *= n[0] * n[1];
      }
    }
  }
  return step;
}

// ================================================== //
// ========== Sinkhorn balancing algorithm ========== //
// ================================================== //
double SinkhornForTensor(Tensor& X, double error_tol, double rep_max, bool verbose) {
  double step = 1.0;
  Int exponent = 0;
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();
  if (verbose) cout << "----- Start Sinkhorn-Knopp Algorithm -----" << endl;
  while (true) {
    for (Int i = 0; i < n1; ++i) {
      for (Int j = 0; j < n2; ++j) {
	double sum = 0.0;
	for (Int k = 0; k < n3; ++k) {
	  sum += X[i][j][k];
	}
	for (Int k = 0; k < n3; ++k) {
	  X[i][j][k] /= (sum * n1 * n2);
	}
      }
    }
    for (Int i = 0; i < n1; ++i) {
      for (Int k = 0; k < n3; ++k) {
	double sum = 0.0;
	for (Int j = 0; j < n2; ++j) {
	  sum += X[i][j][k];
	}
	for (Int j = 0; j < n2; ++j) {
	  X[i][j][k] /= (sum * n1 * n3);
	}
      }
    }
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	double sum = 0.0;
	for (Int i = 0; i < n1; ++i) {
	  sum += X[i][j][k];
	}
	for (Int i = 0; i < n1; ++i) {
	  X[i][j][k] /= (sum * n2 * n3);
	}
      }
    }

    double res = computeResidual(X);
    if (verbose) {
      cout << "Step\t" << step << "\t" << "Residual\t" << res << endl << flush;
    } else {
      if (res < pow(10, -1.0 * (double)exponent)) {
	cout << "  Step ";
	cout << step << ", Residual: " << res << endl;
	exponent++;
      }
    }

    if (res < error_tol) {
      break;
    } else if (step >= rep_max) {
      cout << "  Censored at " << step << " iterations" << endl;
      cout << "  Residual: " << computeResidual(X) << endl;
      break;
    }
    step += 1.0;
  }
  if (verbose) cout << "----- End Sinkhorn-Knopp Algorithm -----" << endl;
  // re-normalization
  sort(n.begin(), n.end(), greater<int>());
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x *= n[0] * n[1];
      }
    }
  }
  return step;
}

// =========================================================== //
// ========== The main function of matrix balancing ========== //
// =========================================================== //
double TensorBalancing(Tensor& X, double error_tol, double rep_max, bool verbose, Int type) {
  if (type == 1) {
    return NewtonBalancing(X, error_tol, rep_max, verbose);
  } else if (type == 2) {
    return SinkhornForTensor(X, error_tol, rep_max, verbose);
  } else {
    return NewtonBalancing(X, error_tol, rep_max, verbose);
  }
}
