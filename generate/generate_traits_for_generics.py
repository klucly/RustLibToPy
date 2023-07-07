from typeinfo import rs_types_to_py, rs_types_to_traits


def rs_to_py_traits(rs_types_to_traits):
    rs_traits_to_py_types = {}
    
    print("Building trait structure for python types")
    for rs_type, py_type in rs_types_to_py.items():
        rs_type = rs_type.replace("-", "")
        traits = rs_types_to_traits[rs_type]

        if py_type in rs_traits_to_py_types.keys():
            rs_traits_to_py_types[py_type].update(traits)
        else:
            rs_traits_to_py_types[py_type] = traits

    print("Collapsing similar traits for python types")

    return rs_traits_to_py_types
    

def save_py_traits():
    py_traits = rs_to_py_traits(rs_types_to_traits)

    with open("py_traits.py", "w") as file:
        file.write(f"py_traits = {py_traits}")


if __name__ == "__main__":
    save_py_traits()