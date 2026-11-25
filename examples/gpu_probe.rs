//! Minimal plumbing probe: does a Metal kernel write anything back at all?
//!
//! Run: cargo run --release --example gpu_probe

fn main() {
    #[cfg(target_os = "macos")]
    {
        let x: Vec<u64> = (0..1024u64).map(|i| i * 3 + 1).collect();
        println!("--- device limits ---");
        {
            use metal::Device;
            let d = Device::system_default().unwrap();
            println!(
                "max_threadgroup_memory = {} bytes, max_threads_per_group = {}",
                d.max_threadgroup_memory_length(),
                d.max_threads_per_threadgroup().width
            );
        }
        println!("--- bisect ---");
        for (name, report) in flashprove::backend::metal::bisect() {
            println!("{name:<16} {report}");
        }
        println!("--- copy probe ---");
        match flashprove::backend::metal::copy_probe(&x) {
            Some(y) => {
                let same = y == x;
                println!("copy_probe: identical={same} first={:?} last={:?}", &y[..4], &y[1018..]);
            }
            None => println!("copy_probe: gpu unavailable"),
        }
    }
}
