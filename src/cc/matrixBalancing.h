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

double EPSILON = 1e-20;

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
  for (Int i = 0; i < X.rows(); i++) {
    for (Int j = 0; j < X.cols(); j++) {
      X(i, j) = data[i][j];
    }
  }
}

// ================================================ //
// ========== Newton balancing algorithm ========== //
// ================================================ //
// preprocess X by a one step Sinkhorn balancing
void preprocess(MatrixXd& X) {
  double sum_vec;
  for (Int i = 0; i < X.rows(); i++) {
    sum_vec = X.row(i).sum();
    for (Int j = 0; j < X.cols(); j++) {
      X(i, j) = X(i, j) / (sum_vec * (double)X.rows());
    }
  }
  for (Int j = 0; j < X.cols(); j++) {
    sum_vec = X.col(j).sum();
    for (Int i = 0; i < X.rows(); i++) {
      X(i, j) = X(i, j) / (sum_vec * (double)X.rows());
    }
  }
}
// pull away zero entries to right-upper and left-lower corners
template<typename V> Int getFirstNonzeroSorting(V&& vec, vector<Int>& idx) {
  for (Int i = 0; i < vec.size(); ++i) {
    if (vec(idx[i]) > EPSILON) return i;
  }
  return -1;
}
template<typename V> Int getFirstNonzero(V&& vec) {
  for (Int i = 0; i < vec.size(); i++) {
    if (vec(i) > EPSILON) return i;
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
void pullZerosSortingRow(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  vector<Int> nonzero_vec;
  vector<Int> zero_vec;

  for (auto&& j : reverse(idx_col)) {
    // sort by j-th col
    for (auto&& i : idx_row) {
      if (X(i, j) > EPSILON)
	nonzero_vec.push_back(i);
      else
	zero_vec.push_back(i);
    }
    idx_row.clear();
    for (auto&& x : nonzero_vec) {
      idx_row.push_back(x);
    }
    for (auto&& x : zero_vec) {
      idx_row.push_back(x);
    }
    nonzero_vec.clear();
    zero_vec.clear();
  }
}
void pullZerosSortingCol(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  vector<Int> nonzero_vec;
  vector<Int> zero_vec;

  for (auto&& i : reverse(idx_row)) {
    // sort by i-th row
    for (auto&& j : idx_col) {
      if (X(i, j) > EPSILON)
	nonzero_vec.push_back(j);
      else
	zero_vec.push_back(j);
    }
    idx_col.clear();
    for (auto&& x : nonzero_vec) {
      idx_col.push_back(x);
    }
    for (auto&& x : zero_vec) {
      idx_col.push_back(x);
    }
    nonzero_vec.clear();
    zero_vec.clear();
  }
}
void pullZerosSorting(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  idx_row.resize(X.rows());
  idx_col.resize(X.cols());
  iota(idx_row.begin(), idx_row.end(), 0);
  iota(idx_col.begin(), idx_col.end(), 0);

  while (!checkPullCondition(X, idx_row, idx_col)) {
    pullZerosSortingRow(X, idx_row, idx_col);
    pullZerosSortingCol(X, idx_row, idx_col);
  }
  MatrixXd X_tmp = X;
  for (Int i = 0; i < X.rows(); ++i) {
    for (Int j = 0; j < X.cols(); ++j) {
      X_tmp(i, j) = X(idx_row[i], idx_col[j]);
    }
  }
  X = X_tmp;
}
// make a node matrix from eigen matrix
void makePosetMatrix(MatrixXd& X, vector<vector<node>>& S, vector<pair<Int, Int>>& idx_tp) {
  Int n = X.rows();
  // initialization
  S.resize(X.rows());
  for (auto&& vec : S) {
    vec.resize(X.cols());
    for (auto&& x : vec){
      x.p = 0;
      x.theta = 0; x.theta_sum = 0; x.theta_sum_prev = 0;
      x.eta = 0;
    }
  }
  // traverse a matrix in the topological order
  for (Int i = 0; i < X.rows() + X.cols() - 1; i++) {
    Int j = i + 1 - X.rows(); // i - j = X.rows() - 1
    if (j < 0) j = 0;
    for (; j <= min(i, (Int)X.cols() - 1); j++) {
      idx_tp.push_back(make_pair(i - j, j));
      S[i - j][j].p = X(i - j, j);
    }
  }
}
// make a beta the submanifold
void makeBeta(vector<vector<node>>& S, vector<Int>& nonzero_row, vector<Int>& nonzero_col, vector<pair<pair<Int, Int>, double>>& beta) {
  beta.push_back(make_pair(make_pair(0, 0), 1.0));
  nonzero_row.resize(S.size());
  nonzero_row[0] = 0;
  for (Int i = 1; i < S.size(); i++) {
    for (Int j = 0; j < S[0].size(); j++) {
      if (S[i][j].p > EPSILON) {
	nonzero_row[i] = j;
	beta.push_back(make_pair(make_pair(i, j), (double)(S.size() - i) / (double)S.size()));
	break;
      }
    }
  }
  nonzero_col.resize(S[0].size());
  nonzero_col[0] = 0;
  for (Int j = 1; j < S[0].size(); j++) {
    for (Int i = 0; i < S.size(); i++) {
      if (S[i][j].p > EPSILON) {
	nonzero_col[j] = i;
	beta.push_back(make_pair(make_pair(i, j), (double)(S[0].size() - j) / (double)S[0].size()));
	break;
      }
    }
  }
}
// compute theta for beta
void computeThetaForBeta(vector<vector<node>>& S, vector<Int>& nonzero_row, vector<Int>& nonzero_col) {
  S[0][0].theta = log(S[0][0].p);
  S[0][0].theta_sum = log(S[0][0].p);
  for (Int i = 1; i < S.size(); i++) {
    S[i][nonzero_row[i]].theta = log(S[i][nonzero_row[i]].p) - S[i - 1][nonzero_row[i - 1]].theta_sum;
    S[i][nonzero_row[i]].theta_sum = S[i][nonzero_row[i]].theta + S[i - 1][nonzero_row[i - 1]].theta_sum;
  }
  for (Int j = 1; j < S[0].size(); j++) {
    S[nonzero_col[j]][j].theta = log(S[nonzero_col[j]][j].p) - S[nonzero_col[j - 1]][j - 1].theta_sum;
    S[nonzero_col[j]][j].theta_sum = S[nonzero_col[j]][j].theta + S[nonzero_col[j - 1]][j - 1].theta_sum;
  }
}
// compute eta for all entries
void computeEta(vector<vector<node>>& S, vector<pair<Int, Int>>& idx_tp) {
  for (auto&& ij : reverse(idx_tp)) {
    double a = S[ij.first][ij.second].p;
    double b = ij.first + 1 < S.size() ? S[ij.first + 1][ij.second].eta : 0;
    double c = ij.second + 1 < S[0].size() ? S[ij.first][ij.second + 1].eta : 0;
    double d = ij.first + 1 < S.size() && ij.second + 1 < S[0].size() ? S[ij.first + 1][ij.second + 1].eta : 0;
    S[ij.first][ij.second].eta = a + b + c - d;
  }
}
// e-projection
void computeP(vector<vector<node>>& S, vector<Int>& nonzero_row, vector<Int>& nonzero_col) {
  S[0][0].theta_sum = S[0][0].theta;
  S[0][0].theta_sum_prev = S[0][0].theta_prev;
  for (Int i = 1; i < S.size(); i++) {
    S[i][nonzero_row[i]].theta_sum = S[i][nonzero_row[i]].theta + S[i - 1][nonzero_row[i - 1]].theta_sum;
    S[i][nonzero_row[i]].theta_sum_prev = S[i][nonzero_row[i]].theta_prev + S[i - 1][nonzero_row[i - 1]].theta_sum_prev;
  }
  for (Int j = 1; j < S[0].size(); j++) {
    S[nonzero_col[j]][j].theta_sum = S[nonzero_col[j]][j].theta + S[nonzero_col[j - 1]][j - 1].theta_sum;
    S[nonzero_col[j]][j].theta_sum_prev = S[nonzero_col[j]][j].theta_prev + S[nonzero_col[j - 1]][j - 1].theta_sum_prev;
  }
  for (Int i = 0; i < S.size(); i++) {
    for (Int j = 0; j < S[0].size(); j++) {
      if (S[i][j].p > EPSILON) {
	double theta_new = S[i][nonzero_row[i]].theta_sum + S[nonzero_col[j]][j].theta_sum - S[0][0].theta_sum;
	double theta_old = S[i][nonzero_row[i]].theta_sum_prev + S[nonzero_col[j]][j].theta_sum_prev - S[0][0].theta_sum_prev;
	S[i][j].p *= exp(theta_new - theta_old);
      }
    }
  }
}
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
void eProject(vector<vector<node>>& S, vector<Int>& nonzero_row, vector<Int>& nonzero_col, vector<pair<pair<Int, Int>, double>>& beta, vector<pair<Int, Int>>& idx_tp) {
  Int S_size = 0;
  for (auto&& vec : S) {
    for (auto&& x : vec) {
      if (x.p > EPSILON) S_size++;
    }
  }
  VectorXd theta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec = VectorXd::Zero(beta.size());
  for (Int i = 0; i < beta.size(); i++) {
    theta_vec[i] = S[beta[i].first.first][beta[i].first.second].theta;
    eta_vec[i]   = S[beta[i].first.first][beta[i].first.second].eta - beta[i].second;
  }
  MatrixXd J(beta.size(), beta.size()); // Jacobian matrix
  for (Int i1 = 0; i1 < beta.size(); i1++) {
    for (Int i2 = 0; i2 < beta.size(); i2++) {
      Int i1_i = beta[i1].first.first;
      Int i1_j = beta[i1].first.second;
      Int i2_i = beta[i2].first.first;
      Int i2_j = beta[i2].first.second;
      J(i1, i2) = S[max(i1_i, i2_i)][max(i1_j, i2_j)].eta;
      J(i1, i2) -= S[i1_i][i1_j].eta * S[i2_i][i2_j].eta * S_size;
    }
  }
  theta_vec += (-1 * J).colPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivLu().solve(eta_vec);

  // store theta
  for (Int i = 0; i < beta.size(); i++) {
    S[beta[i].first.first][beta[i].first.second].theta_prev = S[beta[i].first.first][beta[i].first.second].theta;
    S[beta[i].first.first][beta[i].first.second].theta = theta_vec[i];
  }
  // update p
  computeP(S, nonzero_row, nonzero_col);
  renormalize(S);
  computeEta(S, idx_tp);
}
// compute the residual
double computeResidual(MatrixXd& X) {
  double res = 0.0;
  for (Int i = 0; i < X.rows(); i++)
    res += pow((X.row(i).sum() * (double)X.cols()) - 1.0, 2.0);
  for (Int j = 0; j < X.cols(); j++)
    res += pow((X.col(j).sum() * (double)X.rows()) - 1.0, 2.0);
  return sqrt(res);
}
void recoverZeros(MatrixXd& X, vector<Int>& idx_row, vector<Int>& idx_col) {
  vector<Int> idx_org_row(X.rows());
  iota(idx_org_row.begin(), idx_org_row.end(), 0);
  sort(idx_org_row.begin(), idx_org_row.end(), [&idx_row](Int i, Int j) {return idx_row[i] < idx_row[j];});
  vector<Int> idx_org_col(X.cols());
  iota(idx_org_col.begin(), idx_org_col.end(), 0);
  sort(idx_org_col.begin(), idx_org_col.end(), [&idx_col](Int i, Int j) {return idx_col[i] < idx_col[j];});

  MatrixXd Xnew = X;
  for (Int i = 0; i < X.rows(); i++) {
    Xnew.row(i) = X.row(idx_org_row[i]);
  }
  X = Xnew;
  for (Int j = 0; j < X.cols(); j++) {
    Xnew.col(j) = X.col(idx_org_col[j]);
  }
  X = Xnew;
}
// the main function for newton balancing algorithm
double NewtonBalancing(MatrixXd& X, double error_tol, double rep_max, bool verbose) {
  Int n = X.rows();
  clock_t ts, te;
  vector<Int> idx_row;
  vector<Int> idx_col;
  cout << "  pulling zeros ... " << flush;
  pullZerosSorting(X, idx_row, idx_col);
  cout << "end" << endl;
  // preprocess
  double X_sum = X.sum();
  X /= X_sum;
  preprocess(X);
  // make a node matrix
  vector<vector<node>> S;
  vector<pair<Int, Int>> idx_tp;
  makePosetMatrix(X, S, idx_tp);
  vector<Int> nonzero_row;
  vector<Int> nonzero_col;
  vector<pair<pair<Int, Int>, double>> beta;
  makeBeta(S, nonzero_row, nonzero_col, beta);
  computeThetaForBeta(S, nonzero_row, nonzero_col);
  computeEta(S, idx_tp);

  // run Newton's method
  if (verbose) cout << "----- Start Newton's method -----" << endl;
  double res = 0.0;
  double step = 1.0;
  Int exponent = 0;
  auto t_start = system_clock::now();
  while (step <= rep_max) {
    // perform e-projection
    eProject(S, nonzero_row, nonzero_col, beta, idx_tp);
    // put results to X
    for (Int i = 0; i < X.rows(); i++) {
      for (Int j = 0; j < X.cols(); j++) {
	X(i, j) = S[i][j].p;
      }
    }
    double res_prev = res;
    res = computeResidual(X);
    if (res_prev >= EPSILON && res > res_prev * 100) {
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
  recoverZeros(X, idx_row, idx_col);
  X = X * X.rows();
  return step;
}


// ================================================== //
// ========== Sinkhorn balancing algorithm ========== //
// ================================================== //
double Sinkhorn(MatrixXd& X, double error_tol, double rep_max, bool verbose) {
  double step = 1.0;
  Int exponent = 0;
  if (verbose) cout << endl;
  while (true) {
    for (Int i = 0; i < X.rows(); i++) {
      double sum = X.row(i).sum();
      for (Int j = 0; j < X.cols(); j++) {
	X(i, j) = X(i, j) / (sum * (double)X.rows());
	// X(i, j) = X(i, j) / sum;
      }
    }
    for (Int j = 0; j < X.cols(); j++) {
      double sum = X.col(j).sum();
      for (Int i = 0; i < X.rows(); i++) {
	X(i, j) = X(i, j) / (sum * (double)X.cols());
	// X(i, j) = X(i, j) / sum;
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
double bnewt(MatrixXd& X, double error_tol, double rep_max, bool verbose) {
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
  return step;
}

// =========================================================== //
// ========== The main function of matrix balancing ========== //
// =========================================================== //
double MatrixBalancing(MatrixXd& X, double error_tol, double rep_max, bool verbose, Int type) {
  if (type == 1) {
    return NewtonBalancing(X, error_tol, rep_max, verbose);
  } else if (type == 2) {
    return Sinkhorn(X, error_tol, rep_max, verbose);
  } else if (type == 3) {
    return bnewt(X, error_tol, rep_max, verbose);
  } else {
    return NewtonBalancing(X, error_tol, rep_max, verbose);
  }
}
