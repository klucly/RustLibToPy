from generate_rust import *

if __name__ == "__main__":
    lib_name = "sort"
    runtime_scan = False

    if runtime_scan:
        fn_meta = scan_docs(lib_name)

        with open("buffer.dat", "w") as file:
            file.write(str(fn_meta))

    else:
        with open("buffer.dat", "r") as file:
            fn_meta = eval(file.read())

    # fn_meta = [
    #     (
    #         "l",
    #         "pub fn l<T: Add<(T, K)>, Clone, K>(a: &mut T, b: K)"
    #     ),
    #     (
    #         "a",
    #         "pub fn a<T: Copy, Clone, K: Copy>(b: Vec<(T, K)>, k: K) where T: Add, T: AddAssign"
    #     ),
    #     (
    #         "p",
    #         "pub fn p<T: Add<T, Output = T>, K>(k: &mut Vec<T>) -> &T where T: Copy"
    #     ),
    #     (
    #         "K",
    #         "pub fn K(k: Vec<u32>) -> &u32"
    #     ),
    # ]

    rs_descriptions, py_descriptions = parse_meta(fn_meta)

    rust = generate_rust(lib_name, rs_descriptions)
    python = generate_python(lib_name, py_descriptions)

    print(rust)
    print(python)
