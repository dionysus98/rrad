use std::time::{SystemTime, UNIX_EPOCH};

pub fn rand() -> f32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    // Generate a number from -1 to 1. Adjust '201' and '100' as per your requirement.
    (now % 201) as f32 / 100.0 - 1.0
}
