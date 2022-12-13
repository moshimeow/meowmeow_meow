use color_eyre::Result;
use stereokit::{time::StereoKitTime, StereoKitSettings};
// use stereokit_sys::

fn swing_to_quaternion(swing_x: f32, swing_y: f32) -> glam::Quat {
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

    ret
}

fn lines(sk: &stereokit::lifecycle::StereoKitDraw) {
    
    let p = stereokit::pose::Pose::IDENTITY;
    let time = sk.time_getf();

    let swing_x: f32 = (1.0 - time.sin()) * -0.6;
    let swing_y: f32 = time.cos() * 0.3f32; //time.cos();

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

    sk.run(
        |sk| {
            lines(sk);
        },
        |_| {},
    );

    Ok(())
}
