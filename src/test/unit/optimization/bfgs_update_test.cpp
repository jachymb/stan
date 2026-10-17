#include <gtest/gtest.h>
#include <stan/optimization/bfgs_update.hpp>
#include <cmath>

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_product_form) {
  // the update equals H <- (I - rho s y') tau H (I - rho y s') + rho s s',
  // tau = max(1, s'y / y'Hy), with H = (s'y / y'y) I before the first (reset)
  // update
  const int n = 6;
  int sized = 0;
  stan::optimization::BFGSUpdate_HInv<> bfgsUp;
  Eigen::MatrixXd B(n, n), A, H, I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      B(i, j) = std::sin(1.0 + 3 * i + j);
    }
  }
  A = B.transpose() * B + I;  // a positive definite Hessian
  Eigen::VectorXd s(n), y, g(n), p;
  for (int k = 0; k < 12; ++k) {
    for (int i = 0; i < n; ++i) {
      s[i] = std::sin(1.7 + 2.3 * k + 0.9 * i * i + 0.31 * k * i);
      g[i] = std::cos(0.4 + 1.3 * k * k + 0.7 * i + 0.11 * k * i * i);
    }
    y = A * s;
    const double rho = 1 / y.dot(s);
    const double B0fact = bfgsUp.update(y, s, k == 0);
    const Eigen::MatrixXd V = I - rho * s * y.transpose();
    if (k == 0) {
      EXPECT_FLOAT_EQ(y.squaredNorm() * rho, B0fact);
      H = I / B0fact;
    } else {
      EXPECT_EQ(1.0, B0fact);
      const double tau = y.dot(s) / y.dot(H * y);
      if (tau > 1) {
        H *= tau;
        ++sized;
      }
    }
    H = V * H * V.transpose() + rho * s * s.transpose();
    bfgsUp.search_direction(p, g);
    EXPECT_LE((p + H * g).norm(), 1e-12 * (H * g).norm()) << "update " << k;
  }
  // both branches are exercised
  EXPECT_GT(sized, 0);
  EXPECT_LT(sized, 11);
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_enlarges_small_estimate) {
  // The reset on a pair with curvature 1e4 along e1 sets H = 1e-4 I. The next
  // pair has curvature 1 along e2, so H is first multiplied by 1e4, also
  // along e3, which no pair has explored (H e3 stayed 1e-4 e3 before). A
  // pair with curvature 100 along e3, where H is then 1, does not shrink H:
  // H e1 stays e1, although the first pair measured curvature 1e4 along e1.
  stan::optimization::BFGSUpdate_HInv<> bfgsUp;
  Eigen::VectorXd s = Eigen::VectorXd::Unit(3, 0), p;
  bfgsUp.update(1e4 * s, s, true);
  s = Eigen::VectorXd::Unit(3, 1);
  bfgsUp.update(s, s);
  bfgsUp.search_direction(p, Eigen::VectorXd::Unit(3, 2));
  EXPECT_FLOAT_EQ(-1, p[2]);
  s = Eigen::VectorXd::Unit(3, 2);
  bfgsUp.update(100 * s, s);
  bfgsUp.search_direction(p, Eigen::VectorXd::Unit(3, 0));
  EXPECT_FLOAT_EQ(-1, p[0]);
}

TEST(OptimizationBfgsUpdate, bfgs_update_secant) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  // Construct a set of BFGS update vectors and check that
  // the secant equation H*yk = sk is always satisfied.
  for (int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk, sk, i == 0);

    // Because the constructed update vectors are all orthogonal the secant
    // equation should be exactly satisfied for all nDim updates.
    for (int j = 0; j <= i; j++) {
      sk.setZero(nDim);
      yk.setZero(nDim);
      sk[i - j] = 1;
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir, yk);

      EXPECT_NEAR((sdir + sk).norm(), 0.0, 1e-10);
    }
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_update) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  for (int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk, sk, i == 0);
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_search_direction) {
  const int nDim = 10;
  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);
  sk.setZero(nDim);
  yk.setZero(nDim);
  sk[0] = 1;
  yk[0] = 1;
  bfgsUp.update(yk, sk, true);

  for (int i = 0; i < nDim; i++) {
    for (int j = 0; j <= i; j++) {
      yk.setZero(nDim);
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir, yk);
    }
  }
}
