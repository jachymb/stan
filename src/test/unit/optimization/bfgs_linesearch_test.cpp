#include <gtest/gtest.h>
#include <stan/optimization/bfgs_linesearch.hpp>
#include <stan/math/prim.hpp>
#include <cmath>
#include <stdexcept>

TEST(OptimizationBfgsLinesearch, CubicInterp) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double xMin;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0 * x0 * x0 / 3.0 - x0;
    const double df0 = x0 * x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1 * x1 * x1 / 3.0 - x1;
      const double df1 = x1 * x1 - 1.0;

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -3.0, 2.0);
      EXPECT_NEAR(-3.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -3.0, 0.0);
      EXPECT_NEAR(-3.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -1.0, 2.0);
      EXPECT_NEAR(1.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, 0.0, 2.0);
      EXPECT_NEAR(1.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, 0.5, 1.5);
      EXPECT_NEAR(1.0, xMin, 1e-8);
    }
  }
}

TEST(OptimizationBfgsLinesearch, CubicInterp_6arg) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double x;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0 * x0 * x0 / 3.0 - x0;
    const double df0 = x0 * x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1 * x1 * x1 / 3.0 - x1;
      const double df1 = x1 * x1 - 1.0;

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -3.0 - x0, 2.0 - x0);
      EXPECT_NEAR(-3.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -3.0 - x0, 0.0 - x0);
      EXPECT_NEAR(-3.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -1.0 - x0, 2.0 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, 0.0 - x0, 2.0 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, 0.5 - x0, 1.5 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);
    }
  }
}

class linesearch_testfunc {
 public:
  double operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x) {
    return x.dot(x) - 1.0;
  }
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    f = x.dot(x) - 1.0;
    g = 2.0 * x;
    return 0;
  }
};

TEST(OptimizationBfgsLinesearch, WolfLSZoom) {
  using stan::optimization::WolfLSZoom;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;

  linesearch_testfunc func1;
  Eigen::Matrix<double, -1, 1> x0, x1;
  double f0, f1;
  Eigen::Matrix<double, -1, 1> p, gradx0, gradx1;
  double alpha;
  int ret;

  x0.setOnes(5, 1);
  p = -gradx0;

  func1(x0, f0, gradx0);

  p = -gradx0;

  double dfp = gradx0.dot(p);
  alpha = 2.0;
  x1 = x0 + alpha * p;
  func1(x1, f1, gradx1);

  double dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1, func1, x0, f0, dfp, c1 * dfp,
                   c2 * dfp, p, minAlpha, f0, dfp, alpha, f1, dfp2, 1e-16);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  x1 = x0 + alpha * p;
  func1(x1, f1, gradx1);

  dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1, func1, x0, f0, dfp, c1 * dfp,
                   c2 * dfp, p, minAlpha, f0, dfp, alpha, f1, dfp2, 1e-16);

  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

TEST(OptimizationBfgsLinesearch, wolfeLineSearch) {
  using stan::optimization::WolfeLineSearch;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;
  static const double maxLSIts = 20;
  static const double maxLSRestarts = 10;

  linesearch_testfunc func1;
  Eigen::Matrix<double, -1, 1> x0, x1;
  double f0, f1;
  Eigen::Matrix<double, -1, 1> p, gradx0, gradx1;
  double alpha;
  int ret;

  x0.setOnes(5, 1);
  func1(x0, f0, gradx0);

  p = -gradx0;

  alpha = 2.0;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 0.25;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.25, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

// A 1-D test function for #3229: slope -1 below jump_at, value and slope
// 1e300 from jump_at on, and failed evaluations for fail_lo <= x < fail_hi.
// A line search towards jump_at can bracket the jump, but no point meets
// the Wolfe conditions. The evaluation count is limited, so that an endless
// loop makes a test fail instead of hang.
class linesearch_jumpfunc {
 public:
  linesearch_jumpfunc(double jump_at, double fail_lo, double fail_hi)
      : jump_at_(jump_at), fail_lo_(fail_lo), fail_hi_(fail_hi) {}
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    if (++n_evals_ > 100000)
      throw std::runtime_error("line search does not terminate");
    g.resize(x.size());
    if (x[0] >= fail_lo_ && x[0] < fail_hi_)
      return 1;
    f = x[0] < jump_at_ ? -x[0] : 1e300;
    g[0] = x[0] < jump_at_ ? -1.0 : 1e300;
    return 0;
  }
  int n_evals_ = 0;

 private:
  double jump_at_, fail_lo_, fail_hi_;
};

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_jump_terminates) {
  // The zoom cannot succeed near the jump at 10. It gives up after
  // maxLSIts = 20 evaluations; it used to shrink the bracket down to
  // adjacent doubles (50+ evaluations), where it once looped forever (#3229).
  using stan::optimization::WolfeLineSearch;
  linesearch_jumpfunc func(10.0, 0.0, 0.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << 1.0;
  double alpha = 1.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0,
                                        gradx0, 1e-4, 0.9, 1e-16, 20.0, 10.0));
  EXPECT_EQ(1, ret);
  // 1 evaluation at x0, 2 while expanding, at most 20 in the zoom
  EXPECT_LE(func.n_evals_, 23);
}

TEST(OptimizationBfgsLinesearch, WolfLSZoom_adjacent_bracket) {
  // alo and ahi are adjacent doubles and every trial step rounds to ahi;
  // this looped forever (#3229)
  using stan::optimization::WolfLSZoom;
  linesearch_jumpfunc func(10.0, 0.0, 0.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx1;
  x0 << 0.0;
  p << 1.0;
  const double f0 = 0.0, dfp = -1.0;
  const double alo = std::nextafter(10.0, 0.0), ahi = 10.0;
  double alpha = ahi, f1 = 0.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfLSZoom(alpha, x1, f1, gradx1, func, x0, f0, dfp,
                                   1e-4 * dfp, 0.9 * dfp, p, alo, -alo, -1.0,
                                   ahi, 1e300, 1e300, 1e-16));
  EXPECT_EQ(1, ret);
  EXPECT_LT(func.n_evals_, 1000);
}

TEST(OptimizationBfgsLinesearch, WolfLSZoom_failing_evaluations) {
  // evaluations fail for 10 <= x < 12: the trial steps are halved towards
  // alo = 10 - 1 ulp, reach 10 and then stay there; this looped forever
  // (#3229)
  using stan::optimization::WolfLSZoom;
  linesearch_jumpfunc func(10.0, 10.0, 12.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx1;
  x0 << 0.0;
  p << 1.0;
  const double f0 = 0.0, dfp = -1.0;
  const double alo = std::nextafter(10.0, 0.0), ahi = 13.0;
  double alpha = ahi, f1 = 0.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfLSZoom(alpha, x1, f1, gradx1, func, x0, f0, dfp,
                                   1e-4 * dfp, 0.9 * dfp, p, alo, -alo, -1.0,
                                   ahi, 1e300, 1e300, 1e-16));
  EXPECT_EQ(1, ret);
  EXPECT_LT(func.n_evals_, 1000);
}

TEST(OptimizationBfgsLinesearch, CubicInterp_quadratic_data) {
  // Data from phi(x) = -x + x^2 / (2 m), minimised at m. The cubic
  // coefficient is rounding noise, and the root formula used to lose the
  // minimum: it returned an end point (1e-12 or 1) for x1 = 1 and 3, and
  // missed by up to 28% for x1 = 0.4 (0.128 for m = 0.1).
  using stan::optimization::CubicInterp;
  for (double m : {0.1, 0.25, 0.4, 0.7}) {
    for (double x1 : {0.4, 1.0, 3.0}) {
      const double f1 = -x1 + x1 * x1 / (2 * m);
      const double df1 = -1 + x1 / m;
      EXPECT_NEAR(m, CubicInterp(-1.0, x1, f1, df1, 1e-12, 1.0), 1e-12 * m)
          << "m = " << m << ", x1 = " << x1;
    }
  }
}

// A 1-D piecewise cubic Hermite interpolant of the values f and slopes g at
// the knots x, linear beyond the last knot; counts its evaluations.
class linesearch_hermite {
 public:
  linesearch_hermite(std::vector<double> x, std::vector<double> f,
                     std::vector<double> g)
      : x_(x), f_(f), g_(g) {}
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    ++n_evals_;
    g.resize(1);
    if (x[0] >= x_.back()) {
      f = f_.back() + g_.back() * (x[0] - x_.back());
      g[0] = g_.back();
      return 0;
    }
    size_t k = 0;
    while (x[0] > x_[k + 1]) {
      ++k;
    }
    const double h = x_[k + 1] - x_[k], t = (x[0] - x_[k]) / h;
    f = (2 * t * t * t - 3 * t * t + 1) * f_[k]
        + (t * t * t - 2 * t * t + t) * h * g_[k]
        + (-2 * t * t * t + 3 * t * t) * f_[k + 1]
        + (t * t * t - t * t) * h * g_[k + 1];
    g[0] = (6 * t * t - 6 * t) * (f_[k] - f_[k + 1]) / h
           + (3 * t * t - 4 * t + 1) * g_[k] + (3 * t * t - 2 * t) * g_[k + 1];
    return 0;
  }
  int n_evals_ = 0;

 private:
  std::vector<double> x_, f_, g_;
};

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_armijo_after_expansion) {
  // phi(1) = -2e-4 meets sufficient decrease but not the curvature condition
  // (slope -0.95), so the step is expanded to 10. phi(10) = -3e-4 meets the
  // curvature condition but not sufficient decrease for alpha = 10
  // (-3e-4 > 1e-4 * 10 * -1); it used to be accepted because sufficient
  // decrease was tested for the first trial step.
  using stan::optimization::WolfeLineSearch;
  const double c1 = 1e-4, c2 = 0.9;
  linesearch_hermite func({0.0, 1.0, 10.0}, {0.0, -2e-4, -3e-4},
                          {-1.0, -0.95, -0.5});
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << 1.0;
  double alpha = 1.0;
  int ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1,
                            c2, 1e-16, 20.0, 10.0);
  EXPECT_EQ(0, ret);
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_ascent_direction) {
  // the search refuses an ascent direction without evaluating the function,
  // so that the caller can reset to the gradient (it used to take 50+
  // evaluations to fail)
  using stan::optimization::WolfeLineSearch;
  linesearch_jumpfunc func(10.0, 0.0, 0.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << -1.0;
  double alpha = 1.0;
  int ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0, gradx0,
                            1e-4, 0.9, 1e-16, 20.0, 10.0);
  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, func.n_evals_);
}

// phi(x) = -x + k x^2 / 2, minimised at 1 / k; counts its evaluations
class linesearch_quadratic {
 public:
  explicit linesearch_quadratic(double k) : k_(k) {}
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    ++n_evals_;
    f = -x[0] + 0.5 * k_ * x[0] * x[0];
    g.resize(1);
    g[0] = -1 + k_ * x[0];
    return 0;
  }
  int n_evals_ = 0;

 private:
  double k_;
};

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_stiff_first_step) {
  // The first trial step 1e-3 is 1e8 times the minimiser 1e-11. The cubic
  // trial steps near the low end of the bracket used to be replaced by
  // bisection, so the zoom needed 21 evaluations.
  using stan::optimization::WolfeLineSearch;
  const double c1 = 1e-4, c2 = 0.9;
  linesearch_quadratic func(1e11);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << 1.0;
  double alpha = 1e-3;
  int ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1,
                            c2, 1e-12, 20.0, 10.0);
  EXPECT_EQ(0, ret);
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
  EXPECT_LE(func.n_evals_, 10);
}

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_steps_below_min_alpha) {
  // All acceptable steps lie below minAlpha = 1e-12 (the minimiser is
  // 1e-13), and each first trial step overshoots. The bracket used to start
  // at minAlpha, labelled with the value and slope at 0, and contained no
  // acceptable step.
  using stan::optimization::WolfeLineSearch;
  const double c1 = 1e-4, c2 = 0.9;
  for (double alpha0 : {1e-12, 2e-12, 1e-11}) {
    linesearch_quadratic func(1e13);
    Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
    double f0, f1;
    x0 << 0.0;
    func(x0, f0, gradx0);
    p << 1.0;
    double alpha = alpha0;
    int ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0, gradx0,
                              c1, c2, 1e-12, 20.0, 10.0);
    EXPECT_EQ(0, ret) << "first step " << alpha0;
    EXPECT_LT(alpha, 1e-12) << "first step " << alpha0;
    EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0)) << "first step " << alpha0;
    EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)))
        << "first step " << alpha0;
  }
}

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_tiny_first_step) {
  // The first trial step 2.35e-17 (one unit along a gradient of 4.3e16)
  // overshoots the minimiser 1e-17. The bracket [0, 2.35e-17] is narrower
  // than the fixed width 1e-16 at which the zoom used to give up.
  using stan::optimization::WolfeLineSearch;
  const double c1 = 1e-4, c2 = 0.9;
  linesearch_quadratic func(1e17);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << 1.0;
  double alpha = 2.35e-17;
  int ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1,
                            c2, 1e-12, 20.0, 10.0);
  EXPECT_EQ(0, ret);
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

TEST(OptimizationBfgsLinesearch, WolfLSZoom_max_evals_counts_failures) {
  // every evaluation fails: the zoom gives up after max_evals of them
  using stan::optimization::WolfLSZoom;
  linesearch_jumpfunc func(10.0, 0.0, 20.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx1;
  x0 << 0.0;
  p << 1.0;
  const double f0 = 0.0, dfp = -1.0;
  double alpha = 1.0, f1 = 0.0;
  int ret = WolfLSZoom(alpha, x1, f1, gradx1, func, x0, f0, dfp, 1e-4 * dfp,
                       0.9 * dfp, p, 0.0, f0, dfp, 1.0, 1.0, 1.0, 1e-16, 5.0);
  EXPECT_EQ(1, ret);
  EXPECT_EQ(5, func.n_evals_);
}
