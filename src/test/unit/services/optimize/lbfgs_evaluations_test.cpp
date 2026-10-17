#include <stan/services/optimize/lbfgs.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock_print.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbfgs_evaluations_test {
// the model prints this once per evaluation of the log density
const std::string printed = "log density evaluated";

int count_evaluations(const std::vector<std::string>& logs) {
  int n = 0;
  for (const auto& log : logs) {
    for (auto pos = log.find(printed); pos != std::string::npos;
         pos = log.find(printed, pos + 1)) {
      ++n;
    }
  }
  return n;
}
}  // namespace lbfgs_evaluations_test

struct ServicesOptimizeLbfgsEvaluations : public testing::Test {
  ServicesOptimizeLbfgsEvaluations()
      : init(init_ss), parameter(parameter_ss), model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  stan::test::unit::values_writer parameter;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesOptimizeLbfgsEvaluations, initial_gradient_evaluated_once) {
  // The initialization evaluates the log density (one print) and its
  // gradient (another) at the initial point. The optimizer reuses that
  // gradient as its first gradient evaluation, so the model prints once
  // more than the optimizer's final count of gradient evaluations; it used
  // to print twice more, evaluating the initial gradient again.
  stan::test::unit::instrumented_interrupt interrupt;
  int return_code = stan::services::optimize::lbfgs(
      model, context, 0, 1, 0, 5, 0.001, 1e-12, 10000, 1e-8, 10000000, 1e-8,
      2000, false, 1, interrupt, logger, init, parameter);
  EXPECT_EQ(0, return_code);
  std::vector<std::string> logs = logger.return_all_logs();
  int grad_evals = -1;  // the "# evals" column of the last iteration line
  for (const auto& log : logs) {
    std::istringstream line(log);
    std::vector<std::string> fields{std::istream_iterator<std::string>(line),
                                    std::istream_iterator<std::string>()};
    if (fields.size() >= 7
        && fields[0].find_first_not_of("0123456789") == std::string::npos) {
      grad_evals = std::stoi(fields[6]);
    }
  }
  ASSERT_GT(grad_evals, 0);
  EXPECT_EQ(grad_evals + 1, lbfgs_evaluations_test::count_evaluations(logs));
}

TEST_F(ServicesOptimizeLbfgsEvaluations, initialize_from_value_and_gradient) {
  std::vector<int> disc_vector;
  stan::rng_t rng = stan::services::util::create_rng(0, 1);
  double lp = 0;
  std::vector<double> grad;
  std::vector<double> cont_vector = stan::services::util::initialize<false>(
      model, context, rng, 0, false, logger, init, 100, &lp, &grad);
  ASSERT_EQ(2, grad.size());

  using optimizer_t
      = stan::optimization::BFGSLineSearch<stan_model,
                                           stan::optimization::LBFGSUpdate<>>;
  std::stringstream given_msgs, evaluated_msgs;
  optimizer_t given(model, disc_vector, &given_msgs);
  given.initialize(cont_vector, lp, grad);
  optimizer_t evaluated(model, cont_vector, disc_vector, &evaluated_msgs);
  EXPECT_EQ("", given_msgs.str());
  EXPECT_EQ(1,
            lbfgs_evaluations_test::count_evaluations({evaluated_msgs.str()}));
  EXPECT_EQ(evaluated.logp(), given.logp());
  EXPECT_EQ(evaluated.curr_g(), given.curr_g());
  EXPECT_EQ(evaluated.curr_p(), given.curr_p());
  EXPECT_EQ(evaluated.grad_evals(), given.grad_evals());

  grad[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(given.initialize(cont_vector, lp, grad), std::runtime_error);
}
