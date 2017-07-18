/*
    A Newton balancing algorithm for matrix balancing
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
#include <time.h>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define Int int32_t

using namespace std;
using namespace Eigen;
using namespace std::chrono;

double EPSILON = 1e-300;

// node structure
class node {
public:
  double p;
  double theta, theta_prev, theta_sum, theta_sum_prev;
  double eta, eta_prev;
  double beta;
};

template<typename T> ostream &operator<<(ostream& out, const vector<T>& vec) {
  for (Int i = 0; i < vec.size() - 1; ++i) {
    out << vec[i] << ", ";
  }
  cout << vec[vec.size() - 1];
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
void readFromCSV(MatrixXd& X, ifstream& ifs) {
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
  X = MatrixXd::Zero(data.size(), data[0].size());
  for (Int i = 0; i < X.rows(); ++i) {
    for (Int j = 0; j < X.cols(); ++j) {
      X(i, j) = data[i][j];
    }
  }
}
void outputResidual(double res, double step, Int& exponent, bool verbose) {
  if (verbose) {
    cout << "Step\t" << step << "\t" << "Residual\t" << res << endl << flush;
  } else {
    if (res < pow(10, -1.0 * (double)exponent)) {
      cout << "  Step "; if (step < 10.0) cout << " ";
      cout << step << ", Residual: " << res << endl;
      exponent++;
    }
  }
}
// compute the residual
double computeResidual(vector<vector<node>>& S) {
  Int row = S.size();
  Int col = S[0].size();
  double res = 0.0;
  for (Int i = 0; i < row; ++i) {
    double sum = 0.0;
    for (Int j = 0; j < col; ++j) sum += S[i][j].p;
    res += pow((sum * row) - 1.0, 2.0);
  }
  for (Int j = 0; j < col; ++j) {
    double sum = 0.0;
    for (Int i = 0; i < row; ++i) sum += S[i][j].p;
    res += pow((sum * col) - 1.0, 2.0);
  }
  return sqrt(res);
}
double computeResidual(MatrixXd& X) {
  double res = 0.0;
  for (Int i = 0; i < X.rows(); ++i)
    res += pow((X.row(i).sum() * (double)X.cols()) - 1.0, 2.0);
  for (Int j = 0; j < X.cols(); ++j)
    res += pow((X.col(j).sum() * (double)X.rows()) - 1.0, 2.0);
  return sqrt(res);
}

// ================================================ //
// ========== Newton balancing algorithm ========== //
// ================================================ //
// pull away zero entries to right-upper and left-lower corners
template<typename V> Int getFirstNonzeroSorting(V&& vec, vector<Int>& idx) {
  for (Int i = 0; i < vec.size(); ++i) {
    if (vec(idx[i]) > EPSILON) return i;
  }
  return -1;
}
bool checkPullCondition(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  for (Int i = 0; i < X.rows() - 1; ++i) {
    if (getFirstNonzeroSorting(X.row(idx_row[i]), idx_col) > getFirstNonzeroSorting(X.row(idx_row[i + 1]), idx_col))
      return false;
  }
   for (Int j = 0; j < X.cols() - 1; ++j) {
    if (getFirstNonzeroSorting(X.col(idx_col[j]), idx_row) > getFirstNonzeroSorting(X.col(idx_col[j + 1]), idx_row))
      return false;
  }
  return true;
}
void sortZeroEach(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  vector<pair<Int, Int>> idx_nonzero_row;
  for (auto&& i : idx_row) {
    for (Int j = 0; j < idx_col.size(); ++j) {
      if (X(i, idx_col[j]) > EPSILON) {
	idx_nonzero_row.push_back(make_pair(j, i));
	break;
      }
    }
  }
  stable_sort(idx_nonzero_row.begin(), idx_nonzero_row.end());
  for (Int i = 0; i < idx_row.size(); ++i) {
    idx_row[i] = idx_nonzero_row[i].second;
  }
  vector<pair<Int, Int>> idx_nonzero_col;
  for (auto&& j : idx_col) {
    for (Int i = 0; i < idx_row.size(); ++i) {
      if (X(idx_row[i], j) > EPSILON) {
	idx_nonzero_col.push_back(make_pair(i, j));
	break;
      }
    }
  }
  stable_sort(idx_nonzero_col.begin(), idx_nonzero_col.end());
  for (Int j = 0; j < idx_col.size(); ++j) {
    idx_col[j] = idx_nonzero_col[j].second;
  }
}
void sortZero(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  idx_row.resize(X.rows());
  idx_col.resize(X.cols());
  iota(idx_row.begin(), idx_row.end(), 0);
  iota(idx_col.begin(), idx_col.end(), 0);

  while (!checkPullCondition(X, idx_row, idx_col)) {
    sortZeroEach(X, idx_row, idx_col);
  }

  MatrixXd X_tmp = X;
  for (Int i = 0; i < X.rows(); ++i) {
    for (Int j = 0; j < X.cols(); ++j) {
      X_tmp(i, j) = X(idx_row[i], idx_col[j]);
    }
  }
  X = X_tmp;
}
// preprocess X by a one step Sinkhorn balancing
void preprocess(MatrixXd& X) {
  double sum_vec;
  VectorXd r = VectorXd::Ones(X.rows());
  VectorXd s = VectorXd::Ones(X.cols());
  r = 1.0 / ((X * s).array() * X.rows());
  s = 1.0 / ((X.transpose() * r).array() * X.cols());
  X = r.asDiagonal() * X * s.asDiagonal();
}
// renormalization
void renormalize(vector<vector<node>>& S) {
  // total sum
  double p_sum = 0.0;
  for (auto&& vec : S) {
    for (auto&& x : vec) {
      if (x.p > EPSILON) p_sum += x.p;
    }
  }
  // store the previous theta
  S[0][0].theta_prev = S[0][0].theta;
  // update theta(\bot)
  S[0][0].theta = S[0][0].theta_prev - log(p_sum);
  // update p
  for (auto&& vec : S) {
    for (auto&& x : vec) {
      if (x.p > EPSILON) x.p *= exp(S[0][0].theta - S[0][0].theta_prev);
    }
  }
}
// compute theta
void computeThetaOrg(MatrixXd& X_org, MatrixXd& T) {
  MatrixXd X = X_org;
  X = X / X.sum();
  MatrixXd T_sum = T;
  vector<Int> idx_row(X.rows() - 1);
  vector<Int> idx_col(X.cols() - 1);
  iota(idx_row.begin(), idx_row.end(), 1);
  iota(idx_col.begin(), idx_col.end(), 1);
  T(0, 0) = log(X(0, 0));
  T_sum(0, 0) = log(X(0, 0));
  for (auto&& i : idx_row) {
    T(i, 0) = X(i, 0) > EPSILON ? log(X(i, 0)) - T_sum(i - 1, 0) : 0.0;
    T_sum(i, 0) = T(i, 0) + T_sum(i - 1, 0);
  }
  for (auto&& j : idx_col) {
    T(0, j) = X(0, j) > EPSILON ? log(X(0, j)) - T_sum(0, j - 1) : 0.0;
    T_sum(0, j) = T(0, j) + T_sum(0, j - 1);
  }
  for (auto&& i : idx_row) {
    for (auto&& j : idx_col) {
      T_sum(i, j) = T_sum(i - 1, j) + T_sum(i, j - 1) - T_sum(i - 1, j - 1);
      T(i, j) = X(i, j) > EPSILON ? log(X(i, j)) - T_sum(i, j) : 0.0;
      T_sum(i, j) += T(i, j);
    }
  }
}
// make a node structured matrix S from a given matrix X
void makeNodesFromMatrix(MatrixXd& X, vector<vector<node>>& S) {
  S.resize(X.rows());
  Int i = 0;
  for (auto&& vec : S) {
    vec.resize(X.cols());
    Int j = 0;
    for (auto&& x : vec) {
      x.p = X(i, j);
      x.theta = 0.0; x.theta_sum = 0.0; x.theta_sum_prev = 0.0;
      x.eta = 0.0;
      x.beta = 0.0;
      j++;
    }
    i++;
  }
}
// compute beta (contraint)
void setBeta(vector<vector<node>>& S) {
  Int row = S.size();
  Int col = S[0].size();
  for (Int i = 0; i < row; ++i) {
    for (Int j = 0; j < col; ++j) {
      if (S[i][j].p > EPSILON) {
	S[i][j].beta = (double)(row - i) / (double)row;
	break;
      }
    }
  }
  for (Int j = 0; j < col; ++j) {
    for (Int i = 0; i < row; ++i) {
      if (S[i][j].p > EPSILON) {
	S[i][j].beta = (double)(col - j) / (double)col;
	break;
      }
    }
  }
}
void computeTheta(vector<vector<node>>& S) {
  vector<Int> idx_row(S.size() - 1);
  vector<Int> idx_col(S[0].size() - 1);
  iota(idx_row.begin(), idx_row.end(), 1);
  iota(idx_col.begin(), idx_col.end(), 1);
  S[0][0].theta = log(S[0][0].p);
  S[0][0].theta_sum = log(S[0][0].p);
  for (auto&& i : idx_row) {
    S[i][0].theta = S[i][0].p > EPSILON ? log(S[i][0].p) - S[i - 1][0].theta_sum : 0.0;
    S[i][0].theta_sum = S[i][0].theta + S[i - 1][0].theta_sum;
  }
  for (auto&& j : idx_col) {
    S[0][j].theta = S[0][j].p > EPSILON ? log(S[0][j].p) - S[0][j - 1].theta_sum : 0.0;
    S[0][j].theta_sum = S[0][j].theta + S[0][j - 1].theta_sum;
  }
  for (auto&& i : idx_row) {
    for (auto&& j : idx_col) {
      S[i][j].theta_sum = S[i - 1][j].theta_sum + S[i][j - 1].theta_sum - S[i - 1][j - 1].theta_sum;
      S[i][j].theta = S[i][j].p > EPSILON ? log(S[i][j].p) - S[i][j].theta_sum : 0.0;
      S[i][j].theta_sum += S[i][j].theta;
    }
  }
}
void computeEta(vector<vector<node>>& S) {
  vector<Int> idx_row(S.size() - 1);
  vector<Int> idx_col(S[0].size() - 1);
  iota(idx_row.begin(), idx_row.end(), 0);
  iota(idx_col.begin(), idx_col.end(), 0);
  S[idx_row.size()][idx_col.size()].eta = S[idx_row.size()][idx_col.size()].p;
  for (auto&& i : reverse(idx_row)) {
    S[i][idx_col.size()].eta = S[i][idx_col.size()].p + S[i + 1][idx_col.size()].eta;
  }
  for (auto&& j : reverse(idx_col)) {
    S[idx_row.size()][j].eta = S[idx_row.size()][j].p + S[idx_row.size()][j + 1].eta;
  }
  for (auto&& i : reverse(idx_row)) {
    for (auto&& j : reverse(idx_col)) {
      S[i][j].eta = S[i][j].p + S[i + 1][j].eta + S[i][j + 1].eta - S[i + 1][j + 1].eta;
    }
  }
}
void computeP(vector<vector<node>>& S) {
  vector<Int> idx_row(S.size() - 1);
  vector<Int> idx_col(S[0].size() - 1);
  iota(idx_row.begin(), idx_row.end(), 1);
  iota(idx_col.begin(), idx_col.end(), 1);
  S[0][0].p = exp(S[0][0].theta);
  S[0][0].theta_sum = S[0][0].theta;
  for (auto&& i : idx_row) {
    S[i][0].theta_sum = S[i][0].theta + S[i - 1][0].theta_sum;
    if (S[i][0].p > EPSILON) S[i][0].p = exp(S[i][0].theta_sum);
  }
  for (auto&& j : idx_col) {
    S[0][j].theta_sum = S[0][j].theta + S[0][j - 1].theta_sum;
    if (S[0][j].p > EPSILON) S[0][j].p = exp(S[0][j].theta_sum);
  }
  for (auto&& i : idx_row) {
    for (auto&& j : idx_col) {
      S[i][j].theta_sum = S[i][j].theta + S[i - 1][j].theta_sum + S[i][j - 1].theta_sum - S[i - 1][j - 1].theta_sum;
      if (S[i][j].p > EPSILON) S[i][j].p = exp(S[i][j].theta_sum);
    }
  }
}
void eProject(vector<vector<node>>& S) {
  Int S_size = 0;
  for (auto&& vec : S) {
    for (auto&& x : vec) {
      if (x.beta > EPSILON) S_size++;
    }
  }
  Int row = S.size();
  Int col = S[0].size();
  Int beta_size = row + col - 1;
  VectorXd theta_vec = VectorXd::Zero(beta_size);
  VectorXd eta_vec = VectorXd::Zero(beta_size);
  vector<vector<Int>> idx_vec(beta_size);
  Int i_beta = 0;

  // set beta
  theta_vec[i_beta] = S[0][0].theta;
  eta_vec[i_beta] = S[0][0].eta - S[0][0].beta;
  idx_vec[i_beta].push_back(0);
  idx_vec[i_beta].push_back(0);
  i_beta++;
  for (Int i = 1; i < row; ++i) {
    for (Int j = 0; j < col; ++j) {
      if (S[i][j].beta > EPSILON) {
	theta_vec[i_beta] = S[i][j].theta;
	eta_vec[i_beta] = S[i][j].eta - S[i][j].beta;
	idx_vec[i_beta].push_back(i);
	idx_vec[i_beta].push_back(j);
	i_beta++;
	break;
      }
    }
  }
  for (Int j = 1; j < col; ++j) {
    for (Int i = 0; i < row; ++i) {
      if (S[i][j].beta > EPSILON) {
	theta_vec[i_beta] = S[i][j].theta;
	eta_vec[i_beta] = S[i][j].eta - S[i][j].beta;
	idx_vec[i_beta].push_back(i);
	idx_vec[i_beta].push_back(j);
	i_beta++;
	break;
      }
    }
  }
  // set Jacobian J
  MatrixXd J(beta_size, beta_size); // Jacobian matrix
  for (Int i = 0; i < beta_size; ++i) {
    for (Int j = 0; j < beta_size; ++j) {
      J(i, j)  = S[max(idx_vec[i][0], idx_vec[j][0])][max(idx_vec[i][1], idx_vec[j][1])].eta;
      J(i, j) -= S[idx_vec[i][0]][idx_vec[i][1]].eta * S[idx_vec[j][0]][idx_vec[j][1]].eta;
    }
  }
  theta_vec += (-1 * J).colPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivLu().solve(eta_vec);

  // store theta
  for (Int i = 0; i < beta_size; ++i) {
    S[idx_vec[i][0]][idx_vec[i][1]].theta_prev = S[idx_vec[i][0]][idx_vec[i][1]].theta;
    S[idx_vec[i][0]][idx_vec[i][1]].theta = theta_vec[i];
  }
  // update p
  computeP(S);
  renormalize(S);
  computeEta(S);
}
void computeBalancer(vector<vector<node>>& S, MatrixXd& X, MatrixXd& T, VectorXd& r, VectorXd& s) {
  Int row = r.size();
  Int col = s.size();
  vector<Int> iota_row;
  vector<Int> iota_col;

  for (Int i = 0; i < row; ++i) {
    for (Int j = 0; j < col; ++j) {
      if (S[i][j].beta > EPSILON) {
	iota_row.push_back(j);
	break;
      }
    }
  }
  for (Int j = 0; j < col; ++j) {
    for (Int i = 0; i < row; ++i) {
      if (S[i][j].beta > EPSILON) {
	iota_col.push_back(i);
	break;
      }
    }
  }

  for (Int i = 0; i < row; ++i) {
    r[i] = 0.0;
    for (Int ii = 0; ii <= i; ++ii) {
      r[i] += S[ii][iota_row[ii]].theta - T(ii, iota_row[ii]);
    }
    r[i] = exp(r[i]);
  }
  for (Int j = 0; j < col; ++j) {
    s[j] = 0.0;
    for (Int jj = 0; jj <= j; ++jj) {
      s[j] += S[iota_col[jj]][jj].theta - T(iota_col[jj], jj);
    }
    s[j] = exp(s[j]);
  }

  MatrixXd Xb = r.asDiagonal() * X * s.asDiagonal();
  r *= row / Xb.sum();
}
void recoverZeros(MatrixXd& X, VectorXd& r, VectorXd& s, vector<Int>& idx_row, vector<Int>& idx_col) {
  vector<Int> idx_org_row(X.rows());
  iota(idx_org_row.begin(), idx_org_row.end(), 0);
  sort(idx_org_row.begin(), idx_org_row.end(), [&idx_row](Int i, Int j) {return idx_row[i] < idx_row[j];});
  vector<Int> idx_org_col(X.cols());
  iota(idx_org_col.begin(), idx_org_col.end(), 0);
  sort(idx_org_col.begin(), idx_org_col.end(), [&idx_col](Int i, Int j) {return idx_col[i] < idx_col[j];});

  MatrixXd Xnew = X;
  VectorXd rnew = r;
  VectorXd snew = s;
  for (Int i = 0; i < X.rows(); i++) {
    for (Int j = 0; j < X.cols(); j++) {
      Xnew(i, j) = X(idx_org_row[i], idx_org_col[j]);
    }
  }
  for (Int i = 0; i < X.rows(); i++) {
    rnew[i] = r[idx_org_row[i]];
  }
  for (Int j = 0; j < X.cols(); j++) {
    snew[j] = s[idx_org_col[j]];
  }
  X = Xnew;
  r = rnew;
  s = snew;
}
// the main function for newton balancing algorithm
double NewtonBalancing(MatrixXd& X, VectorXd& r, VectorXd& s, double error_tol, double rep_max, bool verbose) {
  vector<Int> idx_row;
  vector<Int> idx_col;
  // sort rows and orders
  cout << "  sorting rows and columns ... " << flush;
  sortZero(X, idx_row, idx_col);
  cout << "end" << endl;
  MatrixXd T = X; // theta for the input matrix
  computeThetaOrg(X, T);
  // preprocess
  MatrixXd X_proc = X;
  preprocess(X_proc);
  // make a node matrix
  vector<vector<node>> S;
  makeNodesFromMatrix(X_proc, S);
  setBeta(S);
  computeTheta(S);
  computeEta(S);
  // run Newton's method
  if (verbose) cout << "----- Start Newton's method -----" << endl;
  double res = 1.0;
  double res_prev = (double)(X.rows() + X.cols());
  double step = 1.0;
  Int exponent = 0;
  auto t_start = system_clock::now();
  while (step <= rep_max) {
    // perform e-projection
    eProject(S);
    res = computeResidual(S);
    outputResidual(res, step, exponent, verbose); // output the residual
    if (res < error_tol) break;
    if (res_prev >= EPSILON && res > res_prev * 100) {
      cout << "Terminate with failing to convergence." << endl;
      return step;
    }
    res_prev = res;
    step += 1.0;
  }
  if (verbose) cout << "----- End Newton's method -----" << endl;
  // compute balancers
  computeBalancer(S, X, T, r, s);
  // compute the final balanced X
  for (Int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      X(i, j) = S[i][j].p * X.rows();
    }
  }
  // re-sort rows and columns to recover the original row and column orders
  recoverZeros(X, r, s, idx_row, idx_col);
  // MatrixXd Xb = r.asDiagonal() * X * s.asDiagonal();
  // cout << "row sums: " << Xb.rowwise().sum().transpose() << endl;
  // cout << "col sums: " << Xb.colwise().sum() << endl;
  return step;
}

// ================================================== //
// ========== Sinkhorn balancing algorithm ========== //
// ================================================== //
double Sinkhorn(MatrixXd& X, VectorXd& r, VectorXd& s, double error_tol, double rep_max, bool verbose) {
  double step = 1.0;
  Int exponent = 0;
  if (verbose) cout << endl;

  r = VectorXd::Ones(X.rows());
  s = VectorXd::Ones(X.cols());
  // VectorXd r = VectorXd::Ones(X.rows());
  // VectorXd s = VectorXd::Ones(X.cols());
  MatrixXd Xn(X.rows(), X.cols());

  while (true) {
    r = 1.0 / ((X * s).array() * X.rows());
    s = 1.0 / ((X.transpose() * r).array() * X.cols());
    Xn = r.asDiagonal() * X * s.asDiagonal();
    double res = computeResidual(Xn);

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

  r *= X.rows() / Xn.sum();
  X = Xn;
  return step;
}

// ====================================================== //
// ========== The pseudo Newton method (BNEWT) ========== //
// ====================================================== //
void makeSymmetric(MatrixXd& X, MatrixXd& A) {
  MatrixXd Z1 = MatrixXd::Zero(X.rows(), X.rows());
  MatrixXd A1(X.rows(), Z1.cols() + X.cols());
  A1 << Z1, X;
  MatrixXd Z2 = MatrixXd::Zero(X.cols(), X.cols());
  MatrixXd A2(X.cols(), X.rows() + Z2.cols());
  A2 << X.transpose(), Z2;
  A << A1, A2;
}
double bnewt(MatrixXd& X, VectorXd& r, VectorXd& s, double error_tol, double rep_max, bool verbose) {
  double tol = pow(error_tol, 2.0);
  // double tol = error_tol;
  double delta = 0.1;
  double Delta = 3.0;
  Int fl = 0;

  MatrixXd A(X.rows() + X.cols(), X.rows() + X.cols());
  makeSymmetric(X, A);
  Int n = A.rows();
  VectorXd e = VectorXd::Ones(n);
  VectorXd x0 = e;


  Int k = 0;
  double g = 0.9;
  double etamax = 0.1;
  double eta = etamax;
  double eta_o;
  double stop_tol = tol * .5;
  VectorXd x = x0;
  double rt = pow(tol, 2.0);

  VectorXd v = x.cwiseProduct(A * x);
  VectorXd rk = 1.0 - v.array();
  double rho_km1 = rk.transpose() * rk;
  double rout = rho_km1;
  double rold = rout;
  double rat;
  double res_norm;

  double rho_km2;
  double inntertol;
  double alpha;
  double beta;
  double gamma;
  VectorXd y = e;
  VectorXd ynew = e;
  VectorXd Z = rk;
  VectorXd p = rk;
  VectorXd w = rk;
  VectorXd ap = rk;
  VectorXd G = rk;

  Int MVP = 0; // We will count matrix vector products.
  Int i = 0; // Outer iteration count.
  if (fl == 1) {
    cout << "it in. it res" << endl;
  }

  double step = 1.0;
  Int exponent = 0;

  // Outer iteration
  // while (rout > rt) {
  while (step <= rep_max) {
    MatrixXd AS = (x.asDiagonal() * A) * x.asDiagonal();
    double res = 0.0;
    for (Int j = 0; j < X.rows(); j++)
      // res += pow((AS.row(j).sum() * (double)A.cols()) - 1.0, 2.0);
      res += pow(AS.row(j).sum() - 1.0, 2.0);
    for (Int j = X.rows(); j < AS.cols(); j++)
      // res += pow((AS.col(j).sum() * (double)A.rows()) - 1.0, 2.0);
      res += pow(AS.col(j).sum() - 1.0, 2.0);
    res = sqrt(res);
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

    i += 1;
    k = 0;
    y = e;
    inntertol = max(pow(eta, 2.0) * rout, rt);
    while (rho_km1 > inntertol) {
      k += 1;
      if (k == 1) {
	Z = rk.cwiseProduct(v.cwiseInverse());
	p = Z;
	rho_km1 = rk.transpose() * Z;
      } else {
	beta = rho_km1 / rho_km2;
	p = Z + (beta * p);
      }
      // Update search direction efficiently.
      w = x.cwiseProduct(A * x.cwiseProduct(p)) + v.cwiseProduct(p);
      alpha = rho_km1 / (p.transpose() * w);
      ap = alpha * p;
      ynew = y + ap;
      if (ynew.minCoeff() <= delta) {
	if (delta == 0) break;
	G = (delta - y.array()) * ap.cwiseInverse().array();
	gamma = (ap.array() < 0.0).select(G, G.maxCoeff()).minCoeff();
	y = y + gamma * ap;
	break;
      }
      if (ynew.maxCoeff() >= Delta) {
	G = (Delta - y.array()) * ap.cwiseInverse().array();
	gamma = (ynew.array() > Delta).select(G, G.maxCoeff()).minCoeff();
	y = y + gamma * ap;
	break;
      }
      y = ynew;
      rk = rk - alpha * w;
      rho_km2 = rho_km1;
      Z = rk.cwiseProduct(v.cwiseInverse());
      rho_km1 = rk.transpose() * Z;
    }
    x = x.cwiseProduct(y);
    v = x.cwiseProduct(A * x);
    rk = 1.0 - v.array();
    rho_km1 = rk.transpose() * rk;
    rout = rho_km1;
    MVP = MVP + k + 1;
    step = (double)MVP;
    rat = rout / rold;
    rold = rout;
    res_norm = sqrt(rout);
    eta_o = eta;
    eta = g * rat;
    if (g * pow(eta_o, 2.0) > 0.1) {
      eta = max(eta, g * pow(eta_o, 2.0));
    }
    eta = max(min(eta, etamax), stop_tol / res_norm);
    if (fl == 1) {
      cout << "(i, k, res_norm) = (" << i << ", " << k << ", " << res_norm << ")" << endl;
    }
  }

  r = x.head(X.rows());
  s = x.tail(X.cols());
  X = r.asDiagonal() * X * s.asDiagonal();
  return step;
}

// =========================================================== //
// ========== The main function of matrix balancing ========== //
// =========================================================== //
double MatrixBalancing(MatrixXd& X, VectorXd& r, VectorXd& s, double error_tol, double rep_max, bool verbose, Int type) {
  r = VectorXd::Zero(X.rows());
  s = VectorXd::Zero(X.cols());
  switch (type) {
  case 1:
    return NewtonBalancing(X, r, s, error_tol, rep_max, verbose);
  case 2:
    return Sinkhorn(X, r, s, error_tol, rep_max, verbose);
  case 3:
    return bnewt(X, r, s, error_tol, rep_max, verbose);
  default:
    return NewtonBalancing(X, r, s, error_tol, rep_max, verbose);
  }
}
