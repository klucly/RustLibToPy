from typing import Dict, List

try:
    from .trait_type_connection import connection as rs_types_to_traits
    rs_traits_to_types: Dict[str, List[str]]
except:
    print("trait_type_connection.py was not found")
    print("rs_traits_to_types cannot be initialized")
    print("Please run save_traits_for_all to generate")

primitive_docs_link = "https://doc.rust-lang.org/std/primitive.[type].html"

rs_primitive_types = ["array", "bool", "char", "f32", "f64", "i128", "i16", "i32", "i64", "i8", "isize", "slice", "str", "tuple", "u128", "u16", "u32", "u64", "u8", "usize"]
rs_extended_types_docs = {
    "Vec<T>": "https://doc.rust-lang.org/std/vec/struct.Vec.html",
    "String": "https://doc.rust-lang.org/std/string/struct.String.html"
}

# NOTE: If you want to add a new type, add it
# to rs_extended_types_docs with a link to its docs
# from https://doc.rust-lang.org/

# T<U, K, ...> is considered to be a valid name for a type
# ("..." means can be any amount of other or same letters)

rs_extended_types = rs_extended_types_docs.keys()

rs_types_to_py = {
    "array-": "list",
    "bool": "bool",
    "char": "str",
    "f32": "float",
    "f64": "float",
    "i128": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "i8": "int",
    "isize": "int",
    "slice": "bytes", # Only slices of type &[u8] allowed
    "str": "str",
    "String": "str",
    "tuple-": "tuple",
    "u128": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "u8": "int",
    "usize": "int",
    "Vec<T>": "list"
}
# NOTE: Only slices of type &[u8] are allowed
# NOTE: Minus sign after the name means these types
# require special behavior.
# NOTE: Vec<T> is classified to have a valid
# name even with that name having abstract insides
# NOTE: T from the previous example cannot be
# a complex type like T<...>

special_behavior = {
    "array": lambda: None,
    "tuple": lambda: None,
    "slice": lambda: "&[u8]",
    "vector": lambda: None
}
# NOTE: Generics cannot generate arrays and tuples
# since it would need a type for each length of an array
# and for each type of objects of an array making
# everything too complicated


py_types_to_rs = {
    "int": "isize",
    "float": "f64",
    "bool": "bool",
    "str": "&str",
    "bytes": "slice",
    "tuple": "tuple",
    "list": "vector"
}