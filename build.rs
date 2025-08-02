extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/c/rdna_processor.cpp")
        .flag("-mavx2")
        .flag("-mfma")
        .compile("librdna_processor.a");

    println!("cargo::rerun-if-changed=src/c/rdna_processor.c");
    println!("cargo::rustc-link-lib=rdna_processor");
}
