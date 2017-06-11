# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

### Description

The goal of this project is to implement Model Predictive Control to drive the car around the track, using the Udacity Self Driving Car simulator.

![simulator](assets/mpc.jpg)

### Vehicle Model

The vehicle model is based on the kinematic model described in class and based on a bicycle. For simplicity some dynamical effects are ignored, such as inertia, frictionn and torque. The car state is composed by:

- `x`: x coordinate.
- `y`: y coordinate.
- `psi`: heading direction.
- `v`: velocity.
- `cte`: cross-track error.
- `epsi`: orientation error.

The car actuators are two, and are calculated in the method [MPC::Solve](src/MPC.cpp#L165-191) module and returned to [main](src/main.cpp#L131-133):

- `steer`: steering angle
- `throttle`: acceleration (throttle/brake combined)

### Coordinates system

The application receive data from the simulator via websocket message events. The received data is an array of waypoints (x, y) in the global coordinates system, which are transformed to be relative to the car before to be used by the solver. [main.cpp](src/main.cpp#L95-L103)

### Timestep Length and Elapsed Duration (N & dt)

The goal of Model Predictive Control is to optimize the control inputs: [steering_angle, throttle]. An optimizer will tune these inputs until a low cost vector of control inputs is found. The length of this vector is determined by N. After some tests with different values for `N` (time length) `N=10` was selected. During the tests, I observed that large `N` values gives inaccurate prediction values.

MPC attempts to approximate a continues reference trajectory by means of discrete paths between actuations. Larger values of `dt` result in less frequent actuations, which makes it harder to accurately approximate a continuous reference trajectory. During the tests, I observed that small `dt` values makes the car more erratic and large `dt` makes a smoother drive, so the car fails on closed curves. The value `dt=0.15` was finally selected.

N & dt definition at [MPC.cpp](src/MPC.cpp#L14-L15)

### Tunning MPC

Some constants are defined at [MPC.cpp](src/MPC.cpp#L44-L51) to fine tunne the cost function. These constant values should change depending on the `max_velocity` defined at [MPC.cpp](src/MPC.cpp#L42).


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

