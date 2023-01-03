use color_eyre::Result;
use seq_macro::seq;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub, SubAssign};
// use color_eyre::owo_colors::AnsiColors::Default;
use std::default::Default;
// use levenberg_marquardt::LeastSquaresProblem;

use nalgebra;
use stereokit::{time::StereoKitTime, StereoKitSettings};
// use ndarray;

const NUM_LINKS: usize = 2;
const NUM_RESIDUALS: usize = NUM_LINKS * 2;
const NUM_PARAMETERS: usize = NUM_LINKS;

// type magic_matrix = nalgebra::

// Bad!!!!!
// This is because we are going to need to make some big macro that
// instantiates all our LM stuff for various gradient sizes and residual numbers.
// Nova keeps saying const functions might be able to do this
// type OurJacobian = nalgebra::Matrix<f32, NUM_PARAMETERS, NUM_RESIDUALS>;

#[derive(Clone, Copy, Debug)]
struct Jet<const N: usize> {
	val: f32,
	grad: [f32; N],
}

// hey! you probably want num_traits here!
// Sized = "has a constant size at compile time"
trait MeowScalar:
	std::ops::AddAssign
	+ std::ops::SubAssign
	+ std::ops::Add<Output = Self>
	+ std::ops::Sub<Output = Self>
	+ Sized
	+ Copy
	+ Clone
	+ std::fmt::Debug
	+ Default
{
	fn new(val: f32) -> Self;
	fn cos(self) -> Self;
	fn sin(self) -> Self;
}

impl MeowScalar for f32 {
	fn new(val: f32) -> Self {
		return val;
	}
	fn sin(self) -> Self {
		return self.sin();
	}
	fn cos(self) -> Self {
		return self.cos();
	}
}

impl<const N: usize> std::ops::AddAssign for Jet<N> {
	fn add_assign(&mut self, rhs: Self) {
		self.val += rhs.val;
		for i in 0..N {
			self.grad[i] += rhs.grad[i]
		}
	}
}
impl<const N: usize> std::ops::SubAssign for Jet<N> {
	fn sub_assign(&mut self, rhs: Self) {
		self.val -= rhs.val;
		for i in 0..N {
			self.grad[i] -= rhs.grad[i];
		}
	}
}
impl<const N: usize> std::ops::Add for Jet<N> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		let mut s = Self::default();
		s.val = self.val + rhs.val;
		for i in 0..N {
			s.grad[i] = self.grad[i] + rhs.grad[i];
		}
		return s;
	}
}
impl<const N: usize> std::ops::Sub for Jet<N> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		let mut s = Self::default();
		s.val = self.val - rhs.val;
		for i in 0..N {
			s.grad[i] = self.grad[i] - rhs.grad[i];
		}
		return s;
	}
}
impl<const N: usize> Default for Jet<N> {
	fn default() -> Self {
		Self::new(0.0)
	}
}
impl<const N: usize> MeowScalar for Jet<N> {
	fn new(val: f32) -> Self {
		Jet { val, grad: [0.0; N] }
	}

	fn sin(self) -> Self {
		let mut s = Self::default();
		s.val = self.val.sin();

		for i in 0..N {
			s.grad[i] = self.val.cos() * self.grad[i];
		}
		return s;
	}

	fn cos(self) -> Self {
		let mut s = Self::default();
		s.val = self.val.cos();

		for i in 0..N {
			s.grad[i] = -self.val.sin() * self.grad[i];
		}
		return s;
	}
}

trait CostFunctor {
	fn calculate_residual<T: MeowScalar>(
		&mut self,
		input_vector: &[T; NUM_PARAMETERS],
		output_residuals: &mut [T; NUM_RESIDUALS],
	);
}

// fn calc_func_and_jacobian<T: CostFunctor>(
// 	thing: &mut T,
// 	parameters: &[f32; NUM_PARAMETERS],
// 	residuals: &mut[f32; NUM_RESIDUALS],
// 	jacobian: Option<&mut [[f32; NUM_PARAMETERS]; NUM_RESIDUALS]>,
// )

fn calc_func_and_jacobian<T: CostFunctor>(
	thing: &mut T,
	parameters_n: &mut nalgebra::SVector<f32, NUM_PARAMETERS>, //&[f32; NUM_PARAMETERS],
	residuals_n: &mut nalgebra::SVector<f32, NUM_RESIDUALS>,   //[f32; NUM_RESIDUALS],
	// jacobian: Option<&mut [[f32; NUM_PARAMETERS]; NUM_RESIDUALS]>,
	jacobian_n: Option<&mut nalgebra::SMatrix<f32, NUM_RESIDUALS, NUM_PARAMETERS>>,
) {
	let mut parameters: [f32; NUM_PARAMETERS] = Default::default();
	// Slow, verbose copy, to see if it works. Lek pls help!
	for i in 0..NUM_PARAMETERS {
		parameters[i] = parameters_n[i];
	}

	let Some(jacobian_n) = jacobian_n else {

		let mut residuals: [f32; NUM_RESIDUALS] = Default::default();

    thing.calculate_residual(&parameters, &mut residuals);

	for i in 0..NUM_RESIDUALS {
		residuals_n[i] = residuals[i];
	}

    return;
  };

	let mut input_parameters: [Jet<NUM_PARAMETERS>; NUM_PARAMETERS] = Default::default();
	// Initialize the jets. Note, this assumes that Default::default() correctly zeroes everything out.
	// todo Implement Zero in MeowScalar trait
	for i in 0..NUM_PARAMETERS {
		input_parameters[i].val = parameters[i];
		input_parameters[i].grad[i] = 1.0f32;
	}

	let mut output_residuals: [Jet<NUM_PARAMETERS>; NUM_RESIDUALS] = Default::default();

	thing.calculate_residual(&input_parameters, &mut output_residuals);

	// todo I have no idea if this is right, we probably want a ndarray output type
	for r in 0..NUM_RESIDUALS {
		for g in 0..NUM_PARAMETERS {
			jacobian_n[(r, g)] = output_residuals[r].grad[g];
		}
	}
	for i in 0..NUM_RESIDUALS {
		residuals_n[i] = output_residuals[i].val;
	}
}

#[derive(Default)]
struct pgm_state {
	gt_angles: [f32; NUM_LINKS],
	gt_positions: [[f32; 2]; NUM_LINKS],

	recovered_angles: [f32; NUM_LINKS],
	recovered_positions: [f32; NUM_LINKS],
}

// At the end, T is going to be either f32 or Jet, and nothing else. Only at the end.
// We will be instantiating versions of this in _THE SAME FILE_ that can run _RIGHT AFTER EACH OTHER_

fn eval_chain<T: MeowScalar>(angles_in: &[T; NUM_LINKS], pts_out: &mut [[T; 2]; NUM_LINKS]) {
	let mut last_x: T = T::new(0.0f32);
	let mut last_y: T = T::new(0.0f32);

	let mut last_dir: T = T::new(0.0f32);

	for i in 0..NUM_LINKS {
		last_dir += angles_in[i]; //.copy();
		last_x += last_dir.cos();
		last_y += last_dir.sin();

		println!("angles_in[{}]: {:?}", i, angles_in[i]);
		println!("last_dir {:?}", last_dir);
		println!("last_x {:?}", last_x);

		pts_out[i][0] = last_x; //.copy();
		pts_out[i][1] = last_y; //.copy();
	}
}

impl CostFunctor for pgm_state {
	fn calculate_residual<T: MeowScalar>(
		&mut self,
		input_vector: &[T; NUM_LINKS],
		output_residuals: &mut [T; NUM_RESIDUALS],
	) {
		let mut out_pts: [[T; 2]; NUM_LINKS] = Default::default();
		eval_chain(&input_vector, &mut out_pts);

		let mut out_idx: usize = 0;

		println!("Meow! Input:");
		dbg!(input_vector);
		println!("Meow! Output:");
		dbg!(out_pts);

		for i in 0..NUM_LINKS {
			output_residuals[out_idx] = out_pts[i][0] - T::new(self.gt_positions[i][0]);
			out_idx += 1;
			output_residuals[out_idx] = out_pts[i][1] - T::new(self.gt_positions[i][1]);
			out_idx += 1;
		}
	}
}

enum SolverStatus {
	// max_norm |J'(x) * f(x)| < gradient_tolerance
	GRADIENT_TOO_SMALL,
	//  ||dx|| <= parameter_tolerance * (||x|| + parameter_tolerance)
	RELATIVE_STEP_SIZE_TOO_SMALL,
	// cost_threshold > ||f(x)||^2 / 2
	COST_TOO_SMALL,
	// num_iterations >= max_num_iterations
	HIT_MAX_ITERATIONS,
	// (new_cost - old_cost) < function_tolerance * old_cost
	COST_CHANGE_TOO_SMALL,
	// TODO(sameeragarwal): Deal with numerical failures.
}

struct SolverSummary {
	initial_cost_: f32,
	final_cost_: f32,
	gradient_max_norm_: f32,
	iterations_: int,
	status: Status,
}

#[derive(Default)]
struct SolverOptions {
	max_num_iterations: usize,
	gradient_tolerance: f32,
	parameter_tolerance: f32,
	function_tolerance: f32,
	cost_threshold: f32,
	initial_trust_region_radius: f32,
}

impl SolverOptions {
	fn default() -> Self {
		Self {
			max_num_iterations: 50,
			gradient_tolerance: 1e-10,
			parameter_tolerance: 1e-8,
			function_tolerance: 1e-6,
			cost_threshold: std::f32::EPSILON,
			initial_trust_region_radius: 1e4,
		}
	}
}

#[derive(Default)]
struct SolverState<T: CostFunctor> {
	problem_state: T,
	iterations: u32,
	cost_: f32, // = 0.0f32
	gradient_max_norm: f32,
	//
	dx_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	// // so, tiny_solver.hpp uses an externally allocated x
	x: nalgebra::SVector<f32, NUM_PARAMETERS>,
	x_new_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	g_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	jacobi_scaling_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	lm_diagonal_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	lm_step_: nalgebra::SVector<f32, NUM_PARAMETERS>,
	//
	residuals_: nalgebra::SVector<f32, NUM_RESIDUALS>,
	f_x_new_: nalgebra::SVector<f32, NUM_RESIDUALS>,
	//
	jacobian_: nalgebra::SMatrix<f32, NUM_RESIDUALS, NUM_PARAMETERS>,
	jtj_: nalgebra::SMatrix<f32, NUM_PARAMETERS, NUM_PARAMETERS>,
	jtj_regularized_: nalgebra::SMatrix<f32, NUM_PARAMETERS, NUM_PARAMETERS>,
	//
}

fn stupid(
	j: &mut nalgebra::SMatrix<f32, NUM_RESIDUALS, NUM_PARAMETERS>,
	scaling: &nalgebra::SVector<f32, NUM_PARAMETERS>,
) {
	for column in 0..NUM_PARAMETERS {
		let val = scaling[column];
		let new_ = j.column_mut(column) * val;

		// ugh! nalgebra sucks!
		for row in 0..NUM_RESIDUALS {
			j[(row, column)] = new_[row];
		}
	}
}

fn update(s_s: &mut SolverState<pgm_state>) {
	calc_func_and_jacobian(
		&mut s_s.problem_state,
		&mut s_s.x,
		&mut s_s.residuals_,
		Some(&mut s_s.jacobian_),
	);

	if s_s.iterations == 0 {
		for i in 0..NUM_PARAMETERS {
			let guy = s_s.jacobian_.column(i);
			let norm = guy.norm();
			s_s.jacobi_scaling_[i] = 1.0 / (1.0 + norm);
		}
		// s_s.jacobi_scaling_ = 1.0/ (1.0 + s_s.jacobian_.)
	}

	println!("{}", s_s.jacobian_);

	stupid(&mut s_s.jacobian_, &s_s.jacobi_scaling_);
	// for column in 0..NUM_PARAMETERS {
	// 	let val = s_s.jacobi_scaling_[column];
	// 	let new_ = s_s.jacobian_.column_mut(column) * val;
	//
	// 	// ugh! nalgebra sucks!
	// 	for row in 0..NUM_RESIDUALS {
	// 		s_s.jacobian_[(row, column)] = new_[row];
	// 	}
	// 	// s_s.jacobian_.colum
	// 	// s_s.jacobian_.set_column(i, new_);
	// 	// s_s.jacobian_.set_
	// 	// s_s.jacobian_.column_mut(i).mul_s
	// }
	// // s_s.jacobian_ *= s_s.jacobi_scaling_.dia;
	println!("after scaling: {}", s_s.jacobian_);

	s_s.jtj_ = s_s.jacobian_.transpose() * s_s.jacobian_;
	s_s.g_ = s_s.jacobian_.transpose() * s_s.residuals_;

	s_s.gradient_max_norm = s_s.g_.abs().max();
	s_s.cost_ = s_s.residuals_.norm_squared() / 2;
}

fn lmsolve() {
	let mut s_s: SolverState<pgm_state> = Default::default();

	let secret_angles_in: [f32; NUM_LINKS] = Default::default();
	for i in 0..NUM_PARAMETERS {
		s_s.x[i] = -1.2;
	}
	let mut obs_pts_out: [[f32; 2]; NUM_LINKS] = Default::default();

	eval_chain(&secret_angles_in, &mut obs_pts_out);

	s_s.problem_state.gt_angles = secret_angles_in;
	s_s.problem_state.gt_positions = obs_pts_out;

	dbg!("Running update!");

	update(&mut s_s);
	// update(&s_s);
	// update(s_s);

	// let mut cost_: f32 = 0.0f32;
	// let mut dx_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// // so, tiny_solver.hpp uses an externally allocated x
	// let mut x: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// let mut x_new_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// let mut g_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// let mut jacobi_scaling_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// let mut lm_diagonal_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	// let mut lm_step_: nalgebra::SVector<f32, NUM_LINKS> = Default::default();
	//
	// let mut residuals_: nalgebra::SVector<f32, NUM_RESIDUALS> = Default::default();
	// let mut f_x_new_: nalgebra::SVector<f32, NUM_RESIDUALS> = Default::default();
	//
	// let mut jacobian_: nalgebra::SMatrix<f32, NUM_RESIDUALS, NUM_LINKS> = Default::default();
	// let mut jtj_: nalgebra::SMatrix<f32, NUM_LINKS, NUM_LINKS> = Default::default();
	// let mut jtj_regularized_: nalgebra::SMatrix<f32, NUM_LINKS, NUM_LINKS> = Default::default();
}

fn main() -> Result<()> {
	// let angles_in: [f32; NUM_LINKS] = Default::default();
	// let mut pts_out: [[f32; 2]; NUM_LINKS] = Default::default();
	//
	// eval_chain(&angles_in, &mut pts_out);
	//
	// let mut angles_in_jet: [Jet<NUM_PARAMETERS>; NUM_LINKS] = Default::default();
	// for i in 0..NUM_PARAMETERS {
	// 	angles_in_jet[i].val = -1.2;
	// 	angles_in_jet[i].grad[i] = 1.0f32;
	// }
	//
	// let mut pts_out_jet: [[Jet<NUM_PARAMETERS>; 2]; NUM_LINKS] = Default::default();
	//
	// eval_chain(&angles_in_jet, &mut pts_out_jet);
	//
	// println!("{:#?}", angles_in_jet);
	// println!("{:#?}", pts_out_jet);

	lmsolve();

	Ok(())
}
