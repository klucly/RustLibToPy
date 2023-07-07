import requests
from bs4 import BeautifulSoup
from typing import Dict, Iterator, List, Set, Tuple, Optional
from generate.typeinfo import rs_types_to_py, py_types_to_rs, special_behavior
from dataclasses import dataclass
from generate.py_traits import py_traits
from copy import deepcopy

@dataclass
class RsDescription:
    name: str
    args: List[Tuple[str, str, bool]]
    output_type: str
    original_name: str

@dataclass
class PyDescription:
    name: str
    args: List[Tuple[str, str, bool]]
    output_type: str
    supported_types: List[List[str]]

@dataclass
class _RawPyDescription:
    name: str
    args: Set[Tuple[str, str, bool]]
    output_type: Set[str]
    supported_types: List[List[str]]


def remove_not_ascii(inp: str) -> str:
    return bytes([ord(ele) for ele in inp]).decode("utf-8", errors="ignore")

def scan_docs(name: str) -> Set[Tuple[str, str]]:
    docs_link = "https://docs.rs/"+name

    # Request function names
    response = requests.get(docs_link)
    if response.status_code != 200:
        raise ConnectionError(response)
    
    docs = BeautifulSoup(response.content, "html.parser")

    # Docs transfer us to a different
    # link which is much more convenient
    # to use later
    new_link = response.url

    content = docs.find_all(class_ = "fn")
    fn_meta = list()
    for raw_func in content:
        name = raw_func.contents[0]
        link = new_link + raw_func.attrs["href"]

        description_response = requests.get(link)
        if description_response.status_code != 200:
            raise ConnectionError(description_response)
        
        raw_headers = BeautifulSoup(description_response.content, "html.parser")
        header = raw_headers.find(class_ = "rust fn").text

        description = remove_not_ascii(header)

        fn_meta.append((name, description))

    return fn_meta

def _get_args_types(args: List[Tuple[str, str, bool]]) -> Iterator[str]:
    for arg in args:
        yield arg[1]

def _rs_args_to_py(args: List[Tuple[str, str, bool]]) -> Iterator[Tuple[str, str, bool]]:
    for name, type_, is_mut in args:
        yield (name, rs_to_py_type(type_), is_mut)

def _simplify_trait(trait: str) -> str:
    def finish(part: str) -> str:
        end = ""
        for char_ in part:
            if char_ == "[": end = "]" + end
            if char_ == "(": end = ")" + end
            if char_ == "<": end = ">" + end
            if char_ in "])>": end = end[1:]

        return part + end

    main_part = trait.split(",")[0]
    return finish(main_part)

def _is_in_traits(trait: str, traits: List[str]) -> bool:
    def get_depth(trait: str) -> int:
        depth = 0
        max_depth = 0
        for char_ in trait:
            if char_ == "<":
                depth += 1
            elif char_ == ">":
                depth -= 1

            if depth > max_depth: max_depth = depth

        return max_depth
    
    def change_depth(trait: str, max_depth: int) -> str:
        depth = 0
        output = ""
        for char_ in trait:
            if char_ == "<":
                depth += 1
                
            if depth <= max_depth:
                output += char_

            if char_ == ">":
                depth -= 1

        return output

    def simplify(trait: str) -> str:
        simplified = _simplify_trait(trait)

        raw = simplified.split("<")
        for i, part in enumerate(raw):
            if is_generic(part):
                simplified = change_depth(simplified, i-1)

        return simplified

    simplified = simplify(trait)
    main_depth = get_depth(simplified)
    for cmp_trait in traits:
        simplified_cmp = simplify(cmp_trait)
        cmp_depth = get_depth(simplified_cmp)

        min_depth = min(cmp_depth, main_depth)
        if cmp_depth != min_depth:
            simplified_cmp = change_depth(simplified_cmp, min_depth)

        elif main_depth != min_depth:
            simplified = change_depth(simplified, min_depth)
        
        if simplified == simplified_cmp:
            return True
        
    return False

def _get_py_by_traits(traits: str) -> Iterator[str]:
    for type_, cmp_traits in py_traits.items():
        corresponding_rs_type = list(_explicit_py_to_rs_types([type_]))
        if corresponding_rs_type == []:
            continue

        skip = False
        for trait in traits:
            if not _is_in_traits(trait, cmp_traits):
                skip = True
                break

        if skip: continue

        yield type_

def _explicit_py_to_rs_types(py_types: Iterator[str]) -> Iterator[str]:
    for py_type in py_types:
        default_output = py_types_to_rs[py_type]
        if default_output in special_behavior.keys():
            special = special_behavior[default_output]()
            if special:
                yield special
        else:
            yield default_output

def _parse_for_generics(txt: str, generics_map: List[str]) -> str:
    txt = f"<{txt}>"
    special = "[]()<> &,"

    output = ""
    for index, name in enumerate(generics_map):

        left = False
        right = False
        raw = txt.split(name)

        if len(raw) == 1:
            continue

        for i, fragment in enumerate(raw):
            left = fragment[0] in special

            if left and right:
                output += f"@[{index}]" + fragment
            elif i != 0:
                output += name + fragment
            else:
                output = fragment

            right = fragment[-1] in special
        
        txt = output

    if output:
        return output[1:-1]
    else:
        return txt[1:-1]
    
def _replace_generics(type_name: str, index: int, args: List[Tuple[str, str, bool]]) -> None:
    for i, (name, type_, is_mut) in enumerate(args):
        new = type_.replace(f"@[{index}]", type_name)
        args[i] = (name, new, is_mut)

def _replace_arguments(indexes: List[int], generics: List[str], args: List[Tuple[str, str, bool]]) -> None:
    for i_index, index in enumerate(indexes):
        current_generic = generics[i_index][index]
        _replace_generics(current_generic, i_index, args)

def _collapse_generics(
        raw_generics: Dict[str, List[str]],
        raw_args: List[Tuple[str, str, bool]],
        raw_output: str
        ) -> List[List[Tuple[str, str, bool]]]:
    
    # Add output to args to handle it with args
    raw_args.append(("OUTPUT", raw_output, False))
    
    # STEP 1: generate generics map <=> list of generic names
    generics = []
    generics_map = []
    for generic, traits in raw_generics.items():
        generic_types = list(_explicit_py_to_rs_types(_get_py_by_traits(traits)))
        generics.append(generic_types)
        generics_map.append(generic)

    # STEP 2: generate an arg list for simpler use with generics
    # (replace generic names with specific sequence of symbols)
    new_args = []
    for name, type_, is_mut in raw_args:
        new_type = _parse_for_generics(type_, generics_map)
        new_args.append((name, new_type, is_mut))
    
    # STEP 3: generate a list of a list of arguments with generics
    # inserted for each possible reasonable combination of generics
    iterators = [0 for _ in generics]
    lengths = [len(a)-1 for a in generics]
    output = []

    while iterators != lengths:
        buffer_args = deepcopy(new_args)

        for i, iter_ in enumerate(iterators):
            if iter_ == lengths[i]+1:
                iterators[i] = 0
                iterators[i+1] += 1

        _replace_arguments(iterators, generics, buffer_args)
        output.append(buffer_args)

        iterators[0] += 1

        if iterators == lengths:
            _replace_arguments(iterators, generics, new_args)
            output.append(new_args)

    return output

def _build_py_fn_meta(py_fn_meta: List[PyDescription], name: str, args: List[Tuple[str, str, bool]], return_value: str, supported_types: List):
    supported = [list(_get_args_types(a)) for a in supported_types]
    py_fn_meta.append(_RawPyDescription(
        name, set(_rs_args_to_py(args)), set([rs_to_py_type(return_value)]), supported
    ))

def _build_rs_fn_meta(rs_fn_meta: List[PyDescription], name: str, args: List[Tuple[str, str, bool]], return_value: str):
        if args[-1][0] == "OUTPUT":
            cur_args = args[:-1]
        else: cur_args = args

        rs_fn_meta.append(RsDescription(
            generate_fn_name(name, _get_args_types(args)), cur_args, return_value, name
        ))
        
def _recalculate_py_fn_meta(raw: _RawPyDescription, example: List[Tuple[str, str, bool]]) -> Iterator[List[Tuple[str, str, bool]]]:
    args = raw.args
    args.union(raw.output_type)
    arg_dict = {}
    for arg in args:
        if arg[0] not in arg_dict.keys():
            arg_dict[arg[0]] = list(arg[1:])
        else:
            arg_dict[arg[0]][0] += f", {arg[1]}"
    
    arg_list = []
    for example_arg in example:
        name = example_arg[0]
        if "," in arg_dict[name][0]:
            type_ = f"Union[{arg_dict[name][0]}]"
        else:
            type_ = arg_dict[name][0]
        is_mut = arg_dict[name][1]
        arg = (name, type_, is_mut)
        arg_list.append(arg)

    return arg_list

def _build_function_meta(
        rs_fn_meta: List[RsDescription],
        py_fn_meta: List[_RawPyDescription],
        name: str,
        arguments: List[Tuple[str, str, bool]],
        output: str,
        generics: Optional[Dict[str, List[str]]]
        ):
    
    if not generics:
        _build_rs_fn_meta(rs_fn_meta, name, arguments, output)
        py_fn_meta.append(PyDescription(
            name, list(_rs_args_to_py(arguments)), rs_to_py_type(output), [list(_get_args_types(arguments))]
        ))
        return

    data = _collapse_generics(generics, arguments, output)
    for args in data:
        return_value = args[-1][1]
        
        _build_rs_fn_meta(rs_fn_meta, name, args, return_value)
        
        if not py_fn_meta:
            _build_py_fn_meta(py_fn_meta, name, args, return_value, data)
            continue
        
        for description in py_fn_meta:
            if description.name == name:
                description.args = description.args.union(set(_rs_args_to_py(args)))
                description.output_type.add(rs_to_py_type(return_value))
                break

        else:
            _build_py_fn_meta(py_fn_meta, name, args, return_value, data)
            continue

    raw = py_fn_meta.pop(-1)
    meta = _recalculate_py_fn_meta(raw, data[0])
    return_value = meta.pop(-1)[1]
    desk = PyDescription(name, meta, return_value, raw.supported_types)
    py_fn_meta.append(desk)

def is_generic(type_: str) -> bool:
    return type_ == type_.upper()

def _parse_raw_generics(raw_generics: str) -> Dict[str, List[str]]:
    generics = []
    depth = 0
    for generic in raw_generics.split(","):
        depth += generic.count("<")

        if ":" not in generic:
            if depth > 0:
                generics[-1][1][-1] += f", {generic}"
                depth -= generic.count(">")
                continue
                
            if not generics or is_generic(generic):
                generics.append((generic, list()))
                continue

            generics[-1][1].append(generic)
            continue

        depth -= generic.count(">")
        
        current_type, trait = generic.split(":")
        generics_dict = dict(generics)
        if current_type in generics_dict.keys():
            generics_dict[current_type].append(trait)
        else:
            for i, trait_ in enumerate(trait.split("+")):
                if i == 0:
                    generics.append((current_type, [trait_]))
                    continue
                generics[-1][1].append(trait_)

    return dict(generics)

def extract_generics(raw_data: str) -> Dict[str, List[str]]:
    raw_generics, other = raw_data.split(">(")
    other: str

    generics = _parse_raw_generics(raw_generics)
    has_other_generics = other.count("where") == 1

    if has_other_generics:
        raw_other_generics = other.split("where")[1]
        additional_generics = _parse_raw_generics(raw_other_generics)
        for rs_type, traits in additional_generics.items():
            generics[rs_type] += traits
    
    return generics

def extract_output(raw_data: str) -> str:
    if "->" not in raw_data: return "()"
    
    return raw_data.split("->")[1].split("where")[0].strip()

def extract_arguments(raw_data: str) -> List[Tuple[str, str]]:
    # raw_args is now between ...>( and ...-> or ...where
    if ">(" not in raw_data:
        raw_args = raw_data.split("->")[0].split("where")[0][1:]
    else:
        raw_args = raw_data.split(">(")[1].split("->")[0].split("where")[0]
    # Get rid of the last bracket
    raw_args = ")".join(raw_args.split(")")[:-1])

    args = []
    for virtual_arg in raw_args.split(","):
        if ":" not in virtual_arg:
            args[-1] = (args[-1][0], args[-1][1] + f",{virtual_arg}", args[-1][2])
            continue

        arg_name, type_ = virtual_arg.split(":")
        type_ = type_.split()
        is_mut = False

        while "mut" in type_:
            is_mut = True
            type_.remove("mut")
        
        while "&mut" in type_:
            is_mut = True
            type_.remove("&mut")

        type_ = " ".join(type_)

        args.append((arg_name.strip(), type_.strip(), is_mut))

    return args

def _cut_name(name: str, description: str) -> Tuple[str, str]:
    # Get rid of `pub` and `fn` at the beginning 
    while description.split()[0] in ["pub", "fn"]:
        description = " ".join(description.split()[1:])

    # Get everything on the right of the name
    raw_data = name.join(description.split(name)[1:])
    # Simplify
    collapsed_raw_data = raw_data.replace(" ", "")
    collapsed_raw_data = collapsed_raw_data[1:]

    if raw_data[0] not in "<(":
        raise ValueError(f"Inappropriate function ({description})")
    
    return raw_data, collapsed_raw_data

def rs_to_py_type(rs_type: str) -> str:
    rs_type = rs_type.replace("&[u8]", "bytes")
    rs_type = rs_type.replace("[", "array<").replace("<", "[").replace(">","]")
    rs_type = rs_type.replace("&", "").replace("mut", "").replace(" ", "")
    rs_type = rs_type.replace("()", "None").replace("(", "Tuple[").replace(")", "]")
    for cur_rs_type, cur_py_type in rs_types_to_py.items():
        cur_rs_type = cur_rs_type.replace("-", "")
        cur_rs_type = cur_rs_type.split("<")[0]
        rs_type = rs_type.replace(cur_rs_type, cur_py_type)
    
    return rs_type

def generate_fn_name(name: str, types_: Iterator[str]) -> str:
    cur_name = name
    for type_ in types_:
        cur_name += f"_{rs_to_py_type(type_)}"

    for key in "<>[]()&":
        cur_name = cur_name.replace(key, "_")

    return cur_name

def parse_meta(meta: List[Tuple[str, str]]) -> Tuple[List[RsDescription], List[PyDescription]]:
    rs_fn_meta = []
    py_fn_meta = []
    build_fn_meta = lambda name, arguments, output, generics: _build_function_meta(rs_fn_meta, py_fn_meta, name, arguments, output, generics)

    for name, description in meta:
        # Ignore all private function 
        if description.split()[0] != "pub":
            continue
        
        raw_data, collapsed_raw_data = _cut_name(name, description)

        has_generics = raw_data[0] == "<"
        if not has_generics:
            arguments = extract_arguments(raw_data)
            output = extract_output(raw_data)
            build_fn_meta(name, arguments, output, None)
            
            continue
        
        # Skip in case we have a function as an argument
        # too expensive and complex to have functions this way
        if "Fn(" in collapsed_raw_data:
            continue

        generics = extract_generics(collapsed_raw_data)
        arguments = extract_arguments(raw_data)
        output = extract_output(raw_data)
        build_fn_meta(name, arguments, output, generics)
    
    return rs_fn_meta, py_fn_meta

def _bake_outer_type(raw: str) -> str:
    raw = raw.replace("&[u8]", "|BUFFER ZONE|")
    raw = raw.replace("[", "Vec<").replace("]", ">")
    raw = raw.replace("|BUFFER ZONE|", "&[u8]")
    return raw

def _recreate_rs_args(rs_description: RsDescription) -> str:
    output = ""
    for i, arg in enumerate(rs_description.args):
        if i == 0:
            if arg[2]:
                output = f"mut {arg[0]}: {arg[1]}"
                continue

            output = f"{arg[0]}: {arg[1]}"
            continue

        if arg[2]:
            output = f", mut {arg[0]}: {arg[1]}"
            continue

        output += f", {arg[0]}: {arg[1]}"

    output = _bake_outer_type(output)

    return output

def _generate_inner_args(rs_description: RsDescription) -> str:
    output = ""
    for i, arg in enumerate(rs_description.args):
        if i == 0 and arg[2]:
            output = f"&mut {arg[0]}"
            continue
        elif i == 0:
            output = arg[0]
            continue
        
        elif arg[2]:
            output += f", &mut {arg[0]}"
            continue

        output += f", {arg[0]}"

    return output

def _generate_return_type(rs_description: RsDescription) -> str:
    output = f"({rs_description.output_type}, ("
    for i, arg in enumerate(rs_description.args):
        if not arg[2]: continue

        if i == 0:
            output += f"{arg[1]}"
            continue

        output += f", {arg[1]}"
    
    output += ",))"
    output = _bake_outer_type(output)

    return output

def _catch_mut_names(rs_description: RsDescription) -> str:
    output = ""
    for i, arg in enumerate(rs_description.args):
        if not arg[2]: continue

        if i == 0:
            output += f"{arg[0]}"
            continue

        output += f", {arg[0]}"

    return output
    
def generate_rust(lib_name: str, rs_descriptions: List[RsDescription]) -> str:
    output = "#![allow(non_snake_case)]\n"
    output += "use pyo3::prelude::*;\n"
    output += f"use {lib_name}::*;\n\n"

    for rs_description in rs_descriptions:
        name = rs_description.name
        args = _recreate_rs_args(rs_description)
        return_type = _generate_return_type(rs_description)
        output += "#[pyfunction]\n"
        output += f"pub fn {name}({args}) -> PyResult<{return_type}> "
        output += "{\n"
        
        original_name = rs_description.original_name
        inner_args = _generate_inner_args(rs_description)
        mut_names = _catch_mut_names(rs_description)

        output += f"    Ok(({original_name}({inner_args}), ({mut_names},)))\n"
        output += "}\n\n"

    output += "\n#[pymodule]\n"
    output += f"fn X473ycjhOKJH_{lib_name}(_py: Python, m: &PyModule) -> PyResult<()> "
    output += "{\n"
    for rs_description in rs_descriptions:
        output += f"    m.add_function(wrap_pyfunction!({rs_description.name}, m)?)?;\n"
    output += "    Ok(())\n"
    output += "}\n"

    return output

def _recreate_py_args(py_description: PyDescription) -> str:
    output = ""
    for i, arg in enumerate(py_description.args):
        if i == 0:
            output = f"{arg[0]}: {arg[1]}"
            continue

        output += f", {arg[0]}: {arg[1]}"
    
    return output

def generate_python(lib_name: str, py_descriptions: List[PyDescription]) -> str:
    output = "from typing import Union\n\n\n"

    for desc in py_descriptions:
        args = _recreate_py_args(desc)
        output += f"def {desc.name}({args}) -> {desc.output_type}:\n"
        output += f"    from .X473ycjhOKJH_{lib_name} import "

        for i, supported_type in enumerate(desc.supported_types):
            cur_rs_name = generate_fn_name(desc.name, supported_type)
            if i == 0:
                output += f"{cur_rs_name}"
            else:
                output += f", {cur_rs_name}"

        output += "\n"
        output += "    default_muts = ["
        
        i = 0
        for arg in desc.args:
            if arg[2]:
                if i == 0:
                    output += arg[0]
                else:
                    output += f", {arg[0]}"
                i += 1

        output += "]\n    worked = False\n\n"

        for supported_type in desc.supported_types:
            cur_rs_name = generate_fn_name(desc.name, supported_type)
            output += f'''    try:
        return_type, muts = {cur_rs_name}({', '.join(next(zip(*desc.args)))})
        worked = True
    except TypeError: pass\n\n'''
        
        output += "    if not worked: raise TypeError\n\n"

        output += '''    for i, mut in enumerate(muts):
        default_muts[i].clear()
        for a in mut: default_muts[i].append(a)
    return return_type\n\n'''

    return output
