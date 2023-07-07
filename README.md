# RustLibToPy

Allows the use of functions from cargo from rust in python

## Unsupported

* Lifetimes in initialization of functions
* Any kind of sequences except for list by default
* Any functions in local scope

## Use

Make sure you have `maturin` installed first. `pip install maturin`

Run `rustlibtopy.py` with an argument being the name of the library in cargo

Example: `.\rustlibtopy.py sort`

This will generate a folder called `sort` in the example case. To use it
just import it from python and you're done.
