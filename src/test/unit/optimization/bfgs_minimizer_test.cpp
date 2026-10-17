#include <gtest/gtest.h>
#include <stan/optimization/bfgs.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

typedef rosenbrock_model_namespace::rosenbrock_model Model;
typedef stan::optimization::BFGSMinimizer<
    stan::optimization::ModelAdaptor<Model>,
    stan::optimization::BFGSUpdate_HInv<> >
    Optimizer;

class OptimizationBfgsMinimizer : public testing::Test {
 public:
  Eigen::Matrix<double, Eigen::Dynamic, 1> cont_vector;
  std::vector<int> disc_vector;

  void SetUp() {
    cont_vector.resize(2);
    cont_vector[0] = -1;
    cont_vector[1] = 1;
  }
};

TEST_F(OptimizationBfgsMinimizer, constructor) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
}

TEST_F(OptimizationBfgsMinimizer, ls_opts) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs._ls_opts.c1, 1e-4);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.c2, 0.9);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.minAlpha, 1e-12);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.alpha0, 1e-3);
}

TEST_F(OptimizationBfgsMinimizer, conv_opts) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs._conv_opts.maxIts, 10000);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.fScale, 1);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsX, 1e-8);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsF, 1e-12);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsGrad, 1e-8);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolRelF, 1e+4);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolRelGrad, 1e+3);
}

TEST_F(OptimizationBfgsMinimizer, get_qnupdate) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.get_qnupdate();
}

TEST_F(OptimizationBfgsMinimizer, curr_f) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_f(), 4);
}

TEST_F(OptimizationBfgsMinimizer, curr_x) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[1], 1);
}

TEST_F(OptimizationBfgsMinimizer, curr_g) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, curr_p) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_p()[0], 4);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_f) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_f(), 4);
}

TEST_F(OptimizationBfgsMinimizer, prev_x) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.prev_x()[1], 1);
}

TEST_F(OptimizationBfgsMinimizer, prev_g) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.prev_g()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_p) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_p()[0], 0.0040116129);
  EXPECT_FLOAT_EQ(bfgs.prev_p()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_step_size) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_step_size(), 0.0040000002);
}

TEST_F(OptimizationBfgsMinimizer, rel_grad_norm) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.rel_grad_norm(), 0.0012151747);
}

TEST_F(OptimizationBfgsMinimizer, rel_obj_decrease) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.rel_obj_decrease(), 0.0024023936);
}

TEST_F(OptimizationBfgsMinimizer, alpha0) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.alpha0(), 0.001);
}

TEST_F(OptimizationBfgsMinimizer, alpha) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.alpha(), 0.001);
}

TEST_F(OptimizationBfgsMinimizer, iter_num) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.iter_num(), 0);
  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.iter_num(), 1);
}

TEST_F(OptimizationBfgsMinimizer, note) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_TRUE(bfgs.note() == "");
}

TEST_F(OptimizationBfgsMinimizer, get_code_string) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_TRUE(bfgs.get_code_string(0) == "Successful step completed");
  EXPECT_TRUE(
      bfgs.get_code_string(10)
      == "Convergence detected: absolute parameter change was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(20) == "Convergence detected: absolute change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(21) == "Convergence detected: relative change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(30)
              == "Convergence detected: gradient norm is below tolerance");
  EXPECT_TRUE(
      bfgs.get_code_string(31)
      == "Convergence detected: relative gradient magnitude is below "
         "tolerance");
  EXPECT_TRUE(bfgs.get_code_string(40)
              == "Maximum number of iterations hit, may not be at an optima");
  EXPECT_TRUE(bfgs.get_code_string(-1) == "Line search failed to achieve a sufficient decrease, no more progress can be made");
  EXPECT_TRUE(bfgs.get_code_string(42) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(32) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(23) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(94) == "Unknown termination code");
}

TEST_F(OptimizationBfgsMinimizer, initialize) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs.curr_x().size(), 0);
  EXPECT_FLOAT_EQ(bfgs.curr_p().size(), 0);
  EXPECT_FLOAT_EQ(bfgs.curr_g().size(), 0);

  bfgs.initialize(cont_vector);
  EXPECT_FLOAT_EQ(bfgs.curr_x().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[1], 1);
  EXPECT_FLOAT_EQ(bfgs.curr_p().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[0], 4);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[1], 0);
  EXPECT_FLOAT_EQ(bfgs.curr_g().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[1], 0);
  EXPECT_FLOAT_EQ(bfgs.iter_num(), 0);
  EXPECT_TRUE(bfgs.note() == "");
}

TEST_F(OptimizationBfgsMinimizer, step) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);
  bfgs.step();
}

TEST_F(OptimizationBfgsMinimizer, minimize) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs.minimize(cont_vector), 20);
}

TEST_F(OptimizationBfgsMinimizer, initialize_from_value_and_gradient) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> adaptor(rb_model, disc_vector, &out);
  Optimizer evaluated(adaptor), given(adaptor);
  evaluated.initialize(cont_vector);
  given.initialize(cont_vector, evaluated.curr_f(), evaluated.curr_g());
  EXPECT_EQ(evaluated.curr_x(), given.curr_x());
  EXPECT_EQ(evaluated.curr_f(), given.curr_f());
  EXPECT_EQ(evaluated.curr_g(), given.curr_g());
  EXPECT_EQ(evaluated.curr_p(), given.curr_p());
  EXPECT_EQ(0, given.iter_num());

  Eigen::VectorXd g = evaluated.curr_g();
  g[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(given.initialize(cont_vector, evaluated.curr_f(), g),
               std::runtime_error);
  EXPECT_THROW(
      given.initialize(cont_vector, std::numeric_limits<double>::quiet_NaN(),
                       evaluated.curr_g()),
      std::runtime_error);
}

TEST_F(OptimizationBfgsMinimizer, model_adaptor_matches_log_prob_grad) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> adaptor(rb_model, disc_vector, &out);
  for (const auto &xy : {std::make_pair(-1.0, 1.0), std::make_pair(0.5, 2.0),
                         std::make_pair(3.0, -2.0)}) {
    Eigen::VectorXd x(2), g;
    x << xy.first, xy.second;
    double f;
    EXPECT_EQ(0, adaptor(x, f, g));
    std::vector<double> params_r{xy.first, xy.second}, grad;
    double lp = stan::model::log_prob_grad<true, false>(rb_model, params_r,
                                                        disc_vector, grad);
    EXPECT_EQ(-lp, f);
    ASSERT_EQ(2, g.size());
    EXPECT_EQ(-grad[0], g[0]);
    EXPECT_EQ(-grad[1], g[1]);
    EXPECT_EQ(0, stan::math::ChainableStack::instance_->var_stack_.size());
  }
  EXPECT_EQ(3, adaptor.fevals());
}

namespace bfgs_minimizer_test {
// log density x + 10 (with the Jacobian) + 100 (unless propto); mode 1 throws
// std::domain_error, mode 2 std::runtime_error, mode 3 has an infinite
// gradient at 0, mode 4 an infinite value
struct mock_model {
  int mode = 0;
  size_t num_params_r() const { return 1; }
  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T> &x, std::vector<int> &params_i,
             std::ostream *msgs) const {
    if (mode == 1) {
      throw std::domain_error("mock domain error");
    } else if (mode == 2) {
      throw std::runtime_error("mock fatal error");
    } else if (mode == 3) {
      return stan::math::cbrt(x[0]);
    } else if (mode == 4) {
      return x[0] + std::numeric_limits<double>::infinity();
    }
    return x[0] + (jacobian ? 10.0 : 0.0) + (propto ? 0.0 : 100.0);
  }
};
}  // namespace bfgs_minimizer_test

TEST(OptimizationModelAdaptor, template_arguments_and_errors) {
  using bfgs_minimizer_test::mock_model;
  mock_model model;
  std::vector<int> params_i;
  std::stringstream msgs;
  stan::optimization::ModelAdaptor<mock_model, false> adaptor(model, params_i,
                                                              &msgs);
  stan::optimization::ModelAdaptor<mock_model, true> adaptor_jacobian(
      model, params_i, &msgs);
  const auto &stack = stan::math::ChainableStack::instance_->var_stack_;
  Eigen::VectorXd x(1), g;
  x << 0.5;
  double f = 0;
  EXPECT_EQ(0, adaptor(x, f, g));  // propto = true, no Jacobian
  EXPECT_EQ(-0.5, f);
  ASSERT_EQ(1, g.size());
  EXPECT_EQ(-1.0, g[0]);
  EXPECT_EQ(0, adaptor_jacobian(x, f, g));
  EXPECT_EQ(-10.5, f);
  EXPECT_EQ(0, stack.size());

  model.mode = 1;  // a failed evaluation
  EXPECT_EQ(1, adaptor(x, f, g));
  EXPECT_NE(std::string::npos, msgs.str().find("mock domain error"));
  EXPECT_EQ(0, stack.size());
  model.mode = 2;  // other errors propagate
  EXPECT_THROW(adaptor(x, f, g), std::runtime_error);
  EXPECT_EQ(0, stack.size());
  x << 0.0;
  model.mode = 3;
  EXPECT_EQ(3, adaptor(x, f, g));
  EXPECT_EQ(0, stack.size());
  model.mode = 4;
  EXPECT_EQ(2, adaptor(x, f, g));
  EXPECT_EQ(0, stack.size());
}

// f(x) = sum cosh(x_i), minimised at 0; the evaluation fails where cosh
// overflows
struct minimizer_cosh {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    f = x.array().cosh().sum();
    g = x.array().sinh().matrix();
    return std::isfinite(f) && g.allFinite() ? 0 : 1;
  }
};

TEST(OptimizationBfgsMinimizerCosh, large_initial_gradient) {
  // At x0 = (30, 30) the gradient is sinh(30) = 5.3e12 per coordinate. With
  // the default maxStep0 = +inf the first trial step is init_alpha * |g| =
  // 5.3e9, where cosh overflows, and after halving that step 10 times the
  // first line search fails. With maxStep0 = 1 (as the optimize services set
  // it) the first trial moves no coordinate by more than 1.
  stan::optimization::BFGSMinimizer<minimizer_cosh,
                                    stan::optimization::LBFGSUpdate<> >
      lbfgs(minimizer_cosh{});
  lbfgs._ls_opts.maxStep0 = 1;
  Eigen::VectorXd x0(2);
  x0 << 30, 30;
  lbfgs.initialize(x0);
  int ret = lbfgs.step();
  EXPECT_EQ(stan::optimization::TERM_SUCCESS, ret);
  while (ret == stan::optimization::TERM_SUCCESS) {
    ret = lbfgs.step();
  }
  EXPECT_GT(ret, 0);
  EXPECT_NEAR(2, lbfgs.curr_f(), 1e-8);
}

// f(x) = exp(30 x) - 30 x, minimised at 0
struct minimizer_exp {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    f = std::exp(30 * x[0]) - 30 * x[0];
    g.resize(1);
    g[0] = 30 * (std::exp(30 * x[0]) - 1);
    return std::isfinite(f) && g.allFinite() ? 0 : 1;
  }
};

TEST(OptimizationBfgsMinimizerExp, no_relative_convergence_after_reset) {
  // From x0 = 2 (gradient 3e27) the first line search fails with the default
  // maxStep0 = +inf. With the first trial step capped at 1 it accepts x = 1,
  // where the gradient is 3e14 and the inverse Hessian estimate from that one
  // step is 3e-28. The relative gradient test (with CmdStan's default
  // tolerance) must not stop the run there.
  stan::optimization::BFGSMinimizer<minimizer_exp,
                                    stan::optimization::LBFGSUpdate<> >
      lbfgs(minimizer_exp{});
  lbfgs._ls_opts.maxStep0 = 1;
  lbfgs._conv_opts.tolRelGrad = 1e7;
  Eigen::VectorXd x0(1);
  x0 << 2;
  lbfgs.initialize(x0);
  int ret = stan::optimization::TERM_SUCCESS;
  while (ret == stan::optimization::TERM_SUCCESS) {
    ret = lbfgs.step();
  }
  EXPECT_GT(ret, 0);
  EXPECT_NEAR(1, lbfgs.curr_f(), 1e-8);
}

// f(x) = sum_i exp(i x_i) - i x_i, i = 1, ..., n, minimised at 0 with f = n
struct minimizer_exp_sum {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    Eigen::ArrayXd c = Eigen::ArrayXd::LinSpaced(x.size(), 1, x.size());
    Eigen::ArrayXd e = (c * x.array()).exp();
    f = (e - c * x.array()).sum();
    g = (c * (e - 1)).matrix();
    return std::isfinite(f) && g.allFinite() ? 0 : 1;
  }
};

TEST(OptimizationBfgsMinimizerExpSum, dense_bfgs_large_initial_curvature) {
  // From x0 = (5, ..., 5) the curvature falls from 25 e^25 = 1.8e12 to at
  // most 25 at the optimum. The dense inverse Hessian estimate used to keep
  // the scale of the first steps in the directions not explored since, and
  // with CmdStan's default tolerances the relative gradient test stopped the
  // run at f - 5 = 7.6e3.
  stan::optimization::BFGSMinimizer<minimizer_exp_sum,
                                    stan::optimization::BFGSUpdate_HInv<> >
      bfgs(minimizer_exp_sum{});
  bfgs._conv_opts.tolRelGrad = 1e7;
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(5, 5);
  bfgs.initialize(x0);
  int ret = stan::optimization::TERM_SUCCESS;
  while (ret == stan::optimization::TERM_SUCCESS) {
    ret = bfgs.step();
  }
  EXPECT_GT(ret, 0);
  EXPECT_NEAR(5, bfgs.curr_f(), 1e-6);
}
