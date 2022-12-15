use color_eyre::Result;
use seq_macro::seq;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub, SubAssign};
use levenberg_marquardt::LeastSquaresProblem;
use stereokit::{time::StereoKitTime, StereoKitSettings};

const NUM_LINKS: usize = 2;
const NUM_RESIDUALS: usize = NUM_LINKS * 2;
const GRADIENT_SIZE: usize = NUM_LINKS;

// Bad!!!!!
// This is because we are going to need to make some big macro that
// instantiates all our LM stuff for various gradient sizes and residual numbers.
// Nova keeps saying const functions might be able to do this
// type OurJacobian = nalgebra::Matrix<f32, GRADIENT_SIZE, NUM_RESIDUALS>;

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
		Jet {
			val,
			grad: [0.0; N],
		}
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
		input_vector: &[T; NUM_LINKS],
		output_residuals: &mut [T; NUM_RESIDUALS],
	);
}

fn calc_func_and_jacobian<T: CostFunctor>(
	thing: &mut T,
	parameters: &[f32; NUM_LINKS],
	residuals: &mut [f32; NUM_RESIDUALS],
	jacobian: Option<&mut [[f32; GRADIENT_SIZE]; NUM_RESIDUALS]>,
) {
	let Some(jacobian) = jacobian else {
    thing.calculate_residual(parameters, residuals);
    return;
  };

	let mut input_parameters: [Jet<GRADIENT_SIZE>; NUM_LINKS] = Default::default();
	// Initialize the jets. Note, this assumes that Default::default() correctly zeroes everything out.
	// todo Implement Zero in MeowScalar trait
	for i in 0..GRADIENT_SIZE {
		input_parameters[i].val = parameters[i];
		input_parameters[i].grad[i] = 1.0f32;
	}

	let mut output_residuals: [Jet<GRADIENT_SIZE>; NUM_RESIDUALS] = Default::default();

	thing.calculate_residual(&input_parameters, &mut output_residuals);

	// todo I have no idea if this is right, we probably want a ndarray output type
	for r in 0..NUM_RESIDUALS {
		for g in 0..GRADIENT_SIZE {
			jacobian[r][g] = output_residuals[r].grad[g];
		}
	}
}

// struct AutodiffProblem<T: CostFunctor> {
//   state: T,
//
//
// }
//
// impl<T: CostFunctor> LeastSquaresProblem<f32, Const<GRADIENT_SIZE>, Const<NUM_RESIDUALS>> for AutodiffProblem<T> {
//   type ResidualStorage = ();
//   type JacobianStorage = ();
//   type ParameterStorage = ();
//
//   fn set_params(&mut self, x: &nalgebra::base::matrix::Vector<f32, Const<NUM_RESIDUALS>, Self::ParameterStorage>) {
//     todo!()
//   }
//
//   fn params(&self) -> nalgebra::base::matrix::Vector<f32, Const<NUM_RESIDUALS>, Self::ParameterStorage> {
//     todo!()
//   }
//
//   fn residuals(&self) -> Option<nalgebra::base::matrix::Vector<f32, Const<GRADIENT_SIZE>, Self::ResidualStorage>> {
//     todo!()
//   }
//
//   fn jacobian(&self) -> Option<nalgebra::base::matrix::Matrix<f32, Const<GRADIENT_SIZE>, Const<NUM_RESIDUALS>, Self::JacobianStorage>> {
//     todo!()
//   }
// }

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

		for i in 0..NUM_LINKS {
			output_residuals[out_idx] = out_pts[i][0] - T::new(self.gt_positions[i][0]);
			out_idx += 1;
			output_residuals[out_idx] = out_pts[i][1] - T::new(self.gt_positions[i][1]);
			out_idx += 1;
		}
	}
}

fn main() -> Result<()> {
	let angles_in: [f32; NUM_LINKS] = Default::default();
	let mut pts_out: [[f32; 2]; NUM_LINKS] = Default::default();

	eval_chain(&angles_in, &mut pts_out);

	let mut angles_in_jet: [Jet<GRADIENT_SIZE>; NUM_LINKS] = Default::default();
	for i in 0..GRADIENT_SIZE {
		angles_in_jet[i].val = -1.2;
		angles_in_jet[i].grad[i] = 1.0f32;
	}

	let mut pts_out_jet: [[Jet<GRADIENT_SIZE>; 2]; NUM_LINKS] = Default::default();

	eval_chain(&angles_in_jet, &mut pts_out_jet);

	println!("{:#?}", angles_in_jet);
	println!("{:#?}", pts_out_jet);

	Ok(())
}
