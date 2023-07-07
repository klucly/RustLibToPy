import sort
import random
from time import perf_counter
from copy import deepcopy

test1 = [random.random() for _ in range(1000000)]
test2 = deepcopy(test1)
print("Generated")

time1 = perf_counter()
test1.sort()
print(f"Python is {perf_counter()-time1}")

time1 = perf_counter()
sort.quicksort(test2)
print(f"Rust is {perf_counter()-time1}")

assert test1 == test2
