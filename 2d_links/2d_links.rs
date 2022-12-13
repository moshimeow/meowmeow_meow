use std::ops::Add;
// use color_eyre::owo_colors::colors::Default;
use color_eyre::Result;
use stereokit::{time::StereoKitTime, StereoKitSettings};
use seq_macro::seq;

const NUM_LINKS: usize = 2;

struct Jet<const N:usize> {
    val: f32,
    grad: [f32; N]
}

trait MeowScalar {
    fn new(val: f32) -> Self;
}

impl MeowScalar for f32 {
    fn new(val: f32) -> Self {
        val as f32
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
// macro_rules! jet_recursive {
//     ($times:ident) => {
//         seq!{ {N in 0..$times}
//             jet!(N)
//         }
//     };
//     (0) => ();
// }

// seq!(N in (NUM_LINKS*2-1)..NUM_LINKS*2 {
//    jet!(N);
// });
// jet!(NUM_LINKS*2);
jet!({NUM_LINKS*2});
jet!({NUM_LINKS});
// impl MeowScalar for f64 {
//     fn new(val: f64) -> Self {
//         val
//     }
// }


// At the end, T is going to be either f32 or Jet, and nothing else. Only at the end.
// We will be instantiating versions of this in _THE SAME FILE_ that can run _RIGHT AFTER EACH OTHER_
//

// T = f32 OR f64
fn eval_chain<T: MeowScalar>(angles_in: &[T; NUM_LINKS], pts_out: &mut [[T; 2]; NUM_LINKS]) {
    let mut last_x: T = T::new(0.0f32);
    let mut last_y: T = T::new(0.0f32);

    let mut last_dir: T = T::new(0.0f32);

    for i in 0..NUM_LINKS {

    }
}


fn lines(sk: &stereokit::lifecycle::StereoKitDraw) {
    let angles_in: [f32; NUM_LINKS] = Default::default();
    let mut pts_out: [[f32; 2]; NUM_LINKS] = Default::default();

    eval_chain(&angles_in, &mut pts_out);

    let angles_in_jet: [Jet<NUM_LINKS>; NUM_LINKS] = Default::default();
    let mut pts_out_jet: [[Jet<NUM_LINKS>; 2]; NUM_LINKS] = Default::default();


    let x= 10;
    let my_jet: Jet<{ NUM_LINKS * 2 }> = Default::default();
    eval_chain(&angles_in_jet, &mut pts_out_jet);


}

fn main() -> Result<()> {
    let sk = stereokit::StereoKitSettings::default().init()?;

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
