// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

// #include "os/os_time.h"
// #include "util/u_logging.h"

#include "stereokit.h"
#include "stereokit_ui.h"
using namespace sk;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.


// template <typename T> struct Affine2
// {

// }

#define num_links 2

template <typename T>
void
eval_chain(const T in[num_links], T out_pts[num_links][2])
{
	T last_x = {};
	T last_y = {};
	T last_dir = {};


	for (int i = 0; i < num_links; i++) {
		last_dir += in[i];
		last_x += cos(last_dir);
		last_y += sin(last_dir);

		out_pts[i][0] = last_x;
		out_pts[i][1] = last_y;
	}
}

void
eval_chain_simple(double angles[num_links], Eigen::Vector2d out[num_links])
{
	// Paranoia.
	double out_pts[num_links][2];
	eval_chain<double>(angles, out_pts);
	for (int i = 0; i < num_links; i++) {
		out[i].x() = out_pts[i][0];
		out[i].y() = out_pts[i][1];
	}
}

struct pgm_state;

struct CostFunctor
{
	pgm_state *state;

	template <typename T>
	bool
	operator()(const T *const x, T *residual) const;

	CostFunctor(pgm_state *in_state)
	{
		this->state = in_state;
	}
};


struct pgm_state
{
	double gt_angles[num_links] = {};
	Eigen::Vector2d gt_positions[num_links] = {};

	double recovered_angles[num_links];
	Eigen::Vector2d recovered_positions[num_links];
};


template <typename T>
bool
CostFunctor::operator()(const T *const x, T *residual) const
{

	for (int i = 0; i < num_links; i++) {
		std::cout << x[i] << std::endl;
	}

	T out_pts[num_links][2];

	eval_chain(x, out_pts);

  std::cout << std::endl;

  for (int i = 0; i < num_links; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << out_pts[i][j] << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << std::endl;

	for (int i = 0; i < num_links; i++) {
		residual[(i * 2) + 0] = out_pts[i][0] - this->state->gt_positions[i].x();
		residual[(i * 2) + 1] = out_pts[i][1] - this->state->gt_positions[i].y();
	}


	return true;
}



double
do_it(pgm_state &state)
{
	// google::InitGoogleLogging(argv[0]);

	// The variable to solve for with its initial value. It will be
	// mutated in place by the solver.
	// double x[num_links];
	// for ()
	// x[0] = state.recovered_angles[0];
	// x[1] = state.recovered_angles[1];

	// Build the problem.
	Problem problem;

	// Set up the only cost function (also known as residual). This uses
	// auto-differentiation to obtain the derivative (jacobian).
#if 0
	CostFunction *cost_function =
	    new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, num_links * 2, num_links>(new CostFunctor(&state));
#else
	CostFunction *cost_function =
	    new AutoDiffCostFunction<CostFunctor, num_links * 2, num_links>(new CostFunctor(&state));
#endif
	for (int i = 0; i < num_links; i++) {
		state.recovered_angles[i] = -1.2;
	}

	problem.AddResidualBlock(cost_function, nullptr, state.recovered_angles);

	// Run the solver!
	Solver::Options options;
	// options.use_nonmonotonic_steps = true;
	// options.minimizer_type = ceres::LINE_SEARCH;
	options.max_num_iterations = 1000;

	// options.trust_region_strategy_type = ceres::DOGLEG;

	// options.num_threads = 10;
	options.gradient_tolerance = 1e-10;
	options.linear_solver_type = ceres::DENSE_QR;
	// options.linear_solver_type = ceres::SPARSE_SCHUR;
	// options.
	// options.
	// options.minimizer_progress_to_stdout = true;
	Solver::Summary summary;

	// uint64_t start = os_monotonic_get_ns();
	Solve(options, &problem, &summary);
	// uint64_t end = os_monotonic_get_ns();

	// uint64_t diff = end - start;
	// double time_taken = (double)diff / (double)U_TIME_1MS_IN_NS;
	// U_LOG_E("Took %f ms", time_taken);

	// state.recovered_angles[0] = x[0];
	// state.recovered_angles[1] = x[1];

	// // std::cout << summary.BriefReport() << "\n";
	std::cout << summary.FullReport() << "\n";
	// std::cout << "x : "
	//           << " -> " << x[0] << "\n";
	// std::cout << "x : "
	//           << " -> " << x[1] << "\n";


	eval_chain_simple(state.recovered_angles, state.recovered_positions);
	return 0;
}

void
reinit(pgm_state &state)
{
	for (int i = 0; i < num_links; i++) {
		state.recovered_angles[i] = 0;
	}
	eval_chain_simple(state.recovered_angles, state.recovered_positions);
	// eval_chain(state.recovered_angles, state.recovered_positions[0].data(), state.recovered_positions[1].data());
}

// #define step_size 0.01
#define step_size 0.05
bool update_gt = true;


void
step(void *ptr)
{
	pgm_state &state(*(pgm_state *)ptr);

	if (sk::input_key(sk::key_left) & sk::button_state_just_inactive) {
		for (int i = 0; i < num_links; i++) {
			state.gt_angles[i] += step_size * (i + 1);
		}
		// state.gt_angles[0] += 0.01;
		// state.gt_angles[1] += 0.02;
		// state.gt_angles[2] += 0.023;
		update_gt = true;
	} else if (sk::input_key(sk::key_right) & sk::button_state_just_inactive) {
		for (int i = 0; i < num_links; i++) {
			state.gt_angles[i] -= step_size * (i + 1);
		}
		update_gt = true;
	}


	if (update_gt) {
		// U_LOG_E("%f %f", state.gt_angles[0], state.gt_angles[1]);
		reinit(state);
		eval_chain_simple(state.gt_angles, state.gt_positions);
		update_gt = false;
	}



	if (sk::input_key(sk::key_down) & sk::button_state_just_active) {
		do_it(state);
	}



	// std::cout << state.secret_positions[1] << std::endl;
	// std::cout << state.secret_positions[0] << std::endl;

	{
		pose_t p_last = sk::pose_identity;
		p_last.position.z -= 0.1;
		line_add_axis(p_last, 0.1);


		for (int i = 0; i < num_links; i++) {
			pose_t p = sk::pose_identity;
			// std::cout << state.gt_positions << std::endl;
			p.position.x = state.gt_positions[i].x();
			p.position.y = state.gt_positions[i].y();
			p.position.z -= 0.1;


			float hue0 = 0.5;

			line_add(p_last.position, p.position, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
			         color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)), 0.1);
			p_last = p;

			line_add_axis(p, 0.1);
		}
	}

	{
		pose_t p_last = sk::pose_identity;
		line_add_axis(p_last, 0.1);


		for (int i = 0; i < num_links; i++) {
			pose_t p = sk::pose_identity;
			p.position.x = state.recovered_positions[i].x();
			p.position.y = state.recovered_positions[i].y();

			float hue0 = 0.8f;

			line_add(p_last.position, p.position, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
			         color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)), 0.01);
			p_last = p;

			line_add_axis(p, 0.1);
		}
	}
}


void
shutdown(void *ptr)
{}

int
main(int argc, char *argv[])
{


	// u_trace_marker_init();
	sk_settings_t settings = {};
	settings.app_name = "uhhhhhhh??? 🧐🧐🧐🧐🧐";
	settings.assets_folder = "/2/XR/sk-gradient-descent/Assets";
	settings.display_preference = display_mode_flatscreen;
	// settings.display_preference = display_mode_mixedreality;
	settings.disable_flatscreen_mr_sim = true;
	settings.overlay_app = true;
	settings.overlay_priority = 1;
	if (!sk_init(settings))
		return 1;
	sk::render_set_ortho_size(10.5f);
	sk::render_set_projection(sk::projection_ortho);
	sk::render_enable_skytex(false);

	pgm_state stuff; // = {};

	for (int i = 0; i < num_links; i++) {
		stuff.gt_angles[i] = 0.0;
	}

	sk_run_data(step, (void *)&stuff, shutdown, (void *)&stuff);


	return 0;
}