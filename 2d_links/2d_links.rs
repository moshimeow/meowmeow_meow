use std::ops::Add;
use color_eyre::Result;
use stereokit::{time::StereoKitTime, StereoKitSettings};

const NUM_LINKS: usize = 25;

struct Jet<const N: usize> {
    val: f32,
    grad: [f32; N],
}

// struct FWrapper(f32);

//
// impl Add for FWrapper {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         return self.0 + rhs.0;
//     }
// }
// //from trait!
// impl From<f32> for FWrapper {
//     fn from(val: f32) -> Self {
//         FWrapper(val)
//     }
// }
//
// pub trait FFunctions: std::ops::Add + From<f32> + Sized {
//
// }
//
// impl FFunctions for FWrapper {
//
// }
trait MeowScalar {
    fn new(val: f64) -> Self;
}
impl MeowScalar for f32 {
    fn new(val: f64) -> Self{
        val as f32
    }
}


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
    let mut last_x: T = T::new(0.0);
    let mut last_y: T = T::new(0.0);

    let mut last_dir: T = T::new(0.0);

    for i in 0..NUM_LINKS {}
}

fn swing_to_quaternion(swing_x: f32, swing_y: f32) -> glam::Quat {
    let angles_in: [f32; NUM_LINKS] = Default::default();
    let mut pts_out: [[f32; 2]; NUM_LINKS] = Default::default();

    eval_chain(&angles_in, &mut pts_out);

    let mut ret = glam::Quat::default();

    let mut a0 = swing_x;
    let mut a1 = swing_y;
    let theta_squared = a0 * a0 + a1 * a1;

    if theta_squared > f32::EPSILON {
        let theta = theta_squared.sqrt();
        let half_theta = theta * 0.5;
        let k = half_theta.sin() / theta;

        ret.w = half_theta.cos();
        ret.x = a0 * k;
        ret.y = a1 * k;
        ret.z = 0f32;
    } else {
        let k = 0.5;
        ret.w = 1.0;
        ret.x = a0 * k;
        ret.y = a1 * k;
        ret.z = 0.0;
    }

    // glam::Quat<MagicScalar>

    ret
}

fn lines(sk: &stereokit::lifecycle::StereoKitDraw) {
    // sk.
    let p = stereokit::pose::Pose::IDENTITY;
    let time = sk.time_getf();

    let swing_x: f32 = (1.0 - time.sin()) * -0.7;
    let swing_y: f32 = 0.0f32; //time.cos();

    let mut last_quat = glam::Quat::IDENTITY;
    let mut last_pos = glam::Vec3::ZERO;

    let mut pose = stereokit::pose::Pose::new(last_pos, last_quat);
    stereokit::lines::line_add_axis(sk, pose, 0.5);

    for i in 0..3 {
        let mut this_quat = swing_to_quaternion(swing_x, swing_y);
        let mut this_rel_position = glam::Vec3::NEG_Z;
        let mut this_abs_position = glam::Vec3::ZERO;

        this_quat = last_quat * this_quat;

        this_abs_position = last_pos + (this_quat * this_rel_position);

        last_quat = this_quat;
        last_pos = this_abs_position;

        let mut pose = stereokit::pose::Pose::new(last_pos, last_quat);

        stereokit::lines::line_add_axis(sk, pose, 0.5);
    }
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
