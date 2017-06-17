#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// ASSIGN VALUES TO N AND DT.
// For example, if we were to set N to 100, the simulation would run much slower.
// This is because the solver would have to optimize 4 times as many control inputs.
// Ipopt, the solver, permutes the control input values until it finds the lowest cost.
// If you were to open up Ipopt and plot the x and y values as the solver mutates them,
// the plot would look like a worm moving around trying to fit the shape of the reference trajectory.
size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
// It was obtained by measuring the radius formed by running the vehicle in
// the simulator around in a circle with a constant steering angle and
// velocity on a flat terrain.
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
const double Lf = 2.67;

// The solver takes all the state variables and actuator variables in a singular vector
// We establish when one variable starts and another ends to make it easier.
const size_t idx_x = 0;
const size_t idx_y = idx_x + N;
const size_t idx_psi = idx_y + N;
const size_t idx_v = idx_psi + N;
const size_t idx_cte = idx_v + N;
const size_t idx_epsi = idx_cte + N;
const size_t idx_delta = idx_epsi + N;
const size_t idx_a = idx_delta + N - 1;


// Set the targets for cte, epsi, and velocity
// Both the reference cross track and orientation errors are 0.
// the velocity is converted from 50 MPH to meter/second
const double k_ref_cte = 0;
const double k_ref_epsi = 0;
const double k_ref_v =  50 * 0.44704;

// Constants used to fine tunning the cost function.
//
// Some weights used in the cost function, to prioritize some of them on top of
// the others. The final values were obtained by trial and error.
//
// For the reference state costs, Staying on track and maintaining the steering
// minimum error is important, so 100 is given to cte and epsi.
// A 1 is given to velocity, to accelerate and break as necessary to stay on track.
const int k_w_cte = 10;
const int k_w_epsi = 10;
const int k_w_v = 1;
// For the actuators, 100 is assigned to the steering actuator,
// to have a small value between timesteps.
// 1 is given to acceleration, to allow the vehicle to accelerate or break as necessary.
const int k_w_current_delta = 100;
const int k_w_current_a = 1;
// For the sequential actuators, 100 is assigned to the steering angle change,
// to smooth steering inputs.
// 1 is given to acceleration, to allow the vehicle to accelerate or break as necessary.
const int k_w_diff_delta = 100;
const int k_w_diff_a = 1;

class FG_eval {
 private:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs_;

 public:

  FG_eval(Eigen::VectorXd coeffs) {
    coeffs_ = coeffs;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  void operator()(ADvector& fg, const ADvector& vars) {
    // COSTS
    // vars is the vector of variables and fg is the vector of constraints.

    // Since 0 is the index at which Ipopt expects fg to store the cost value,
    // we sum all the components of the cost and store them at index 0.
    fg[0] = 0;

    // The part of the cost based on the reference state.
    // In each iteration through the loop, we sum three components
    // to reach the aggregate cost: our cross-track error,
    // our heading error, and our velocity error.
    for (int i = 0; i < N; i++) {
      fg[0] += k_w_cte * CppAD::pow(vars[idx_cte + i] - k_ref_cte, 2);
      fg[0] += k_w_epsi * CppAD::pow(vars[idx_epsi + i] - k_ref_epsi, 2);
      fg[0] += k_w_v * CppAD::pow(vars[idx_v + i] - k_ref_v, 2);
    }

    // We've already taken care of the main objective - to minimize
    // our cross track, heading, and velocity errors.
    //
    // A further enhancement is to constrain erratic control inputs.
    //
    // Minimize the use of actuators. For example, if we're making a
    // turn, we'd like the turn to be smooth, not sharp. Additionally,
    // the vehicle velocity should not change too radically.
    for (int i = 0; i < N - 1; i++) {
      fg[0] += k_w_current_delta * CppAD::pow(vars[idx_delta + i], 2);
      fg[0] += k_w_current_a * CppAD::pow(vars[idx_a + i], 2);
    }

    // Minimize the value gap between sequential actuations.
    // The goal of this final loop is to make control decisions
    // more consistent, or smoother. The next control input
    // should be similar to the current one.
    for (int i = 0; i < N - 2; i++) {
      fg[0] += k_w_diff_delta * CppAD::pow(vars[idx_delta + i + 1] - vars[idx_delta + i], 2);
      fg[0] += k_w_diff_a * CppAD::pow(vars[idx_a + i + 1] - vars[idx_a + i], 2);
    }

    // CONSTRAINTS

    // initial constraints.
    // fg[0] stores the cost value, so there's always an offset of 1.
    // So fg[1 + idx_psi] is where we store the initial value of ψ.
    fg[1 + idx_x] = vars[idx_x];
    fg[1 + idx_y] = vars[idx_y];
    fg[1 + idx_psi] = vars[idx_psi];
    fg[1 + idx_v] = vars[idx_v];
    fg[1 + idx_cte] = vars[idx_cte];
    fg[1 + idx_epsi] = vars[idx_epsi];

    // The rest of the constraints
    for (int i = 0; i < N - 1; i++) {

      // The state at time t+1.
      AD<double> x1 = vars[idx_x + i + 1];
      AD<double> y1 = vars[idx_y + i + 1];
      AD<double> psi1 = vars[idx_psi + i + 1];
      AD<double> v1 = vars[idx_v + i + 1];
      AD<double> cte1 = vars[idx_cte + i + 1];
      AD<double> epsi1 = vars[idx_epsi + i + 1];

      // The state at time t.
      AD<double> x0 = vars[idx_x + i];
      AD<double> y0 = vars[idx_y + i];
      AD<double> psi0 = vars[idx_psi + i];
      AD<double> v0 = vars[idx_v + i];
      AD<double> cte0 = vars[idx_cte + i];
      AD<double> epsi0 = vars[idx_epsi + i];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[idx_delta + i];
      AD<double> a0 = vars[idx_a + i];

      AD<double> f0 = coeffs_[0] + coeffs_[1] * x0 + coeffs_[2] * CppAD::pow(x0, 2);// + coeffs_[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x0); // + 3 * coeffs_[3] * CppAD::pow(x0, 2));

      AD<double> psi_offset = v0 / Lf * delta0 * dt;

      fg[2 + idx_x + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[2 + idx_y + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[2 + idx_psi + i] = psi1 - (psi0 + psi_offset);
      fg[2 + idx_v + i] = v1 - (v0 + a0 * dt);
      fg[2 + idx_cte + i] = cte1 - (f0 - y0 + v0 * epsi0 * dt);
      fg[2 + idx_epsi + i] = epsi1 - (psi0 - psides0 + psi_offset);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}

MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // for readibility purposes
  const double x = state[0];
  const double y = state[1];
  const double psi = state[2];
  const double v = state[3];
  const double cte = state[4];
  const double epsi = state[5];

  // Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  // 4 * 10 + 2 * 9
  const int k_dim_state = 6;
  const int k_dim_actuators = 2;
  const int k_num_vars = N * k_dim_state + (N - 1) * k_dim_actuators;
  // Set the number of constraints
  const int k_num_constraints = N * k_dim_state;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(k_num_vars);
  for (int i = 0; i < k_num_vars; i++) {
    vars[i] = 0.0;
  }

  // Set the initial variable values
  vars[idx_x] = x;
  vars[idx_y] = y;
  vars[idx_psi] = psi;
  vars[idx_v] = v;
  vars[idx_cte] = cte;
  vars[idx_epsi] = epsi;

  Dvector vars_lowerbound(k_num_vars);
  Dvector vars_upperbound(k_num_vars);

  // SET LOWER AND UPPER LIMITS FOR VARIABLES.
  // Set all non-actuators upper and lowerlimits to the max negative and positive values.
  for (int i = 0; i < idx_delta; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }
  // The upper and lower limits of delta are set to -25
  // and 25 degrees (values in radians).
  const double k_delta_max = 25 * M_PI / 180.0;
  for (int i = idx_delta; i < idx_a; i++) {
    vars_lowerbound[i] = -k_delta_max;
    vars_upperbound[i] = k_delta_max;
  }
  // Acceleration/decceleration upper and lower limits.
  for (int i = idx_a; i < k_num_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(k_num_constraints);
  Dvector constraints_upperbound(k_num_constraints);
  for (int i = 0; i < k_num_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[idx_x] = x;
  constraints_lowerbound[idx_y] = y;
  constraints_lowerbound[idx_psi] = psi;
  constraints_lowerbound[idx_v] = v;
  constraints_lowerbound[idx_cte] = cte;
  constraints_lowerbound[idx_epsi] = epsi;

  constraints_upperbound[idx_x] = x;
  constraints_upperbound[idx_y] = y;
  constraints_upperbound[idx_psi] = psi;
  constraints_upperbound[idx_v] = v;
  constraints_upperbound[idx_cte] = cte;
  constraints_upperbound[idx_epsi] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // To deal with the 100ms latency, we return the
  // [delta_start + 1] and [a_start + 1] values,
  // because is just 1 timestep interval `dt=0.1s`.
  vector<double> result = {solution.x[idx_delta + 1], solution.x[idx_a + 1]};

  // The rest of the values are used to display the green line.
  x_vals_.clear();
  y_vals_.clear();
  for (int i = 0; i < N - 1; i++) {
    x_vals_.push_back(solution.x[idx_x + 1 + i]);
    y_vals_.push_back(solution.x[idx_y + 1 + i]);
  }

  return result;
}
