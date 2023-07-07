from typeinfo import *
from typing import Set
import requests
from bs4 import BeautifulSoup

def get_traits_for_type(type_: str, docs_link = None) -> Set[str]:
    if not docs_link:
        response = requests.get(primitive_docs_link.replace("[type]", type_))
    else:
        response = requests.get(docs_link)

    soup = BeautifulSoup(response.content, "html.parser")
    sidebar = soup.find(class_ = "sidebar-elems")
    raw_contents = sidebar.contents[0].contents
    for i, element in enumerate(raw_contents):
        if len(element.contents) == 1 and element.contents[0].attrs["href"] == "#trait-implementations":
            raw_trait_list = raw_contents[i+1]

    traits = set()
    for raw_trait in raw_trait_list:
        traits.add(raw_trait.text)
    
    return traits

def save_traits_for_all():
    build = {}
    for rs_type in rs_primitive_types:
        print(f"Requesting info for {rs_type}")
        traits = get_traits_for_type(rs_type)
        build[rs_type] = traits

    for rs_type, link in rs_extended_types_docs.items():
        print(f"Requesting info for {rs_type}")
        traits = get_traits_for_type(rs_type, link)
        build[rs_type] = traits

    with open("trait_type_connection.py", "w") as file:
        file.write(f"connection = {build}")
        
if __name__ == "__main__":
    save_traits_for_all()
