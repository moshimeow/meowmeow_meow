use std::ops::{Add, AddAssign};
// use color_eyre::owo_colors::colors::Default;
use color_eyre::Result;
use stereokit::{time::StereoKitTime, StereoKitSettings};
use seq_macro::seq;

const NUM_LINKS: usize = 2;
const RESIDUAL_SIZE_DEFAULT: usize = NUM_LINKS * 2;

struct Jet<const N:usize> {
    val: f32,
    grad: [f32; N]

    // fn cos()
}



// Sized = "has a constant size at compile time"
trait MeowScalar: AddAssign + Sized + Cosine {
    fn new(val: f32) -> Self;
}

trait Cosine {
    fn cos(self) -> Self;
}


impl Cosine for f32 {
    fn cos(self) -> Self {
        self.cos()
    }
}

impl MeowScalar for f32 {
    fn new(val: f32) -> Self {
        val as f32
    }
}

impl Cosine for Jet<RESIDUAL_SIZE_DEFAULT> {
    fn cos(self) -> Self {

        // let mut stored_grad = self.grad;
        // self.val = self.val.cos()


        let mut s = Self {
            val: self.val.cos(),
            grad: std::default::Default()
        };

        for i in 0..RESIDUAL_SIZE_DEFAULT {
            s.grad[i] = -self.grad[i].sin()
        }

        s
    }
}

macro_rules! jet {
    ($num:expr) => {
        impl MeowScalar for Jet<$num> {
            fn new(val: f32) -> Self {
                Jet {
                    val,
                    grad: Default::default(),
                }
            }
        }
        impl Default for Jet<$num> {
            fn default() -> Self {
                Self::new(0.0)
            }
        }

    }
}

jet!({RESIDUAL_SIZE_DEFAULT});


// At the end, T is going to be either f32 or Jet, and nothing else. Only at the end.
// We will be instantiating versions of this in _THE SAME FILE_ that can run _RIGHT AFTER EACH OTHER_
//

// T = f32 OR f64
fn eval_chain<T: MeowScalar>(angles_in: &[T; NUM_LINKS], pts_out: &mut [[T; 2]; NUM_LINKS]) {
    let mut last_x: T = T::new(0.0f32);
    let mut last_y: T = T::new(0.0f32);

    let mut last_dir: T = T::new(0.0f32);

    for i in 0..NUM_LINKS {
        last_dir += angles_in[i].copy();
        last_x += last_dir.cos();
        last_y += last_dir.sin();

        pts_out[i][0] = last_x.copy();
        pts_out[i][1] = last_y.copy();
    }
}


fn lines(sk: &stereokit::lifecycle::StereoKitDraw) {
    let angles_in: [f32; NUM_LINKS] = Default::default();
    let mut pts_out: [[f32; 2]; NUM_LINKS] = Default::default();

    eval_chain(&angles_in, &mut pts_out);

    let angles_in_jet: [Jet<RESIDUAL_SIZE_DEFAULT>; NUM_LINKS] = Default::default();
    let mut pts_out_jet: [[Jet<RESIDUAL_SIZE_DEFAULT>; 2]; NUM_LINKS] = Default::default();


    let x= 10;
    let my_jet: Jet< RESIDUAL_SIZE_DEFAULT > = Default::default();
    eval_chain(&angles_in_jet, &mut pts_out_jet);


}

fn main() -> Result<()> {
    let sk = stereokit::StereoKitSettings::default().init()?;

    let mut sdk = 0.0f32;
    sdk = sdk.cos();
    println!(sdk);


    unsafe {
        stereokit_sys::render_set_projection(stereokit_sys::projection__projection_ortho);
    }
    sk.run(
        |sk| {
            lines(sk);
        },
        |_| {},
    );

    Ok(())
}
