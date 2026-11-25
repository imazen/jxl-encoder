//! Diagnose reference execution from the same target runtime as cross tests.
fn main() {
    let tool = std::env::var("DJXL_PATH").expect("DJXL_PATH must select the pinned decoder");
    eprintln!("tool={tool} metadata={:?}", std::fs::metadata(&tool));
    match std::process::Command::new(&tool).arg("--version").output() {
        Ok(output) => {
            eprintln!("status={:?}", output.status);
            eprintln!("stdout={}", String::from_utf8_lossy(&output.stdout));
            eprintln!("stderr={}", String::from_utf8_lossy(&output.stderr));
            assert!(output.status.success());
            assert!(
                [&output.stdout, &output.stderr].into_iter().any(|bytes| {
                    let banner = String::from_utf8_lossy(bytes);
                    let mut tokens = banner.split_ascii_whitespace();
                    tokens.next() == Some("djxl") && tokens.next() == Some("v0.13.0")
                }),
                "expected the libjxl v0.13.0 reference"
            );
        }
        Err(error) => panic!("reference spawn failed: {error:?}"),
    }
}

#[test]
fn reference_runs_from_a_test_worker() {
    main();
}
