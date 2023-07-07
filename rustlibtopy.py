from sys import argv
from os import system, listdir
from generate_rust import scan_docs, parse_meta, generate_rust, generate_python
libname = argv[1]

print("Fetching data...")
meta = scan_docs(libname)
print("Parsing meta...")
rs_desc, py_desc = parse_meta(meta)
print("Generating code...")
main_rs = generate_rust(libname, rs_desc)
main_py = generate_python(libname, py_desc)

print("Creating a project...")
system(f"maturin new X473ycjhOKJH_{libname} -b pyo3")
with open(f"X473ycjhOKJH_{libname}\\src\\lib.rs", "w") as file:
    file.write(main_rs)

with open(f"X473ycjhOKJH_{libname}\\cargo.toml", "a") as file:
    file.write(f'{libname} = "*"\n')

print("Generating an environment...")
if ".venv" not in listdir("."):
    system("python -m venv .env")
    system(f"move .\\.env .\\.venv")

print("Compiling...")
system(f"maturin develop --release -m .\\X473ycjhOKJH_{libname}\\Cargo.toml")

if libname in listdir("."):
    system(f"rmdir /s /q {libname}")

system(f"move .\\.venv\\Lib\\site-packages\\X473ycjhOKJH_{libname} .\\{libname}")

with open(f"{libname}\\__init__.py", "w") as file:
    file.write(main_py)

print("\n\x1B[32mDone\033[0m\t\t\n")
