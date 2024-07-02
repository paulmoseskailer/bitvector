# Advanced Data Structures Project

Implementation of a bitvector supporting access, rank and select queries for the Advanced Data Structures lecture in SS 2024.

## Instructions

### Requirements

* Rust 

Tested on Ubuntu 20.04 with 
`cargo 1.78.0 (54d8815d0 2024-03-26)`.
See [here](https://www.rust-lang.org/tools/install) for rust install instructions.

### How to Compile

Clone the repository, run `cargo build` in the same directory as this README file.

### How to Run

```
cargo run input_file output_file
```

(this also compiles if needed). Alternatively, after [compiling](#how-to-compile), you can run the equivalent

```
./target/debug/bitvector input_file output_file
```

Note that running can take a long time.
This is due to the creation of all the support datastructures (especially for select support).
For example, processing an input bit vector of length $2^{27}$ (~16MB) takes 2.5 minutes on my i5-7600 with 16GB RAM, while processing 1000 random queries on that vector takes ~1ms total.

### Input format

As specified in the project requirements, the input has to be a text file containing

1. An integer $n \in \mathbb{N}$ in the first line,
2. the bitvector as a string over the alphabet $\{0,1\}$ followed by
3. $n$ lines of one `access i`,`rank b i` or `select b i` query each.

Where `b` $\in \{0,1\}$ and `i` is a valid index.
If the indices are invalid, undefined behaviour may occur.

### Ouptut format

The specified `output_file` contains $n$ lines with the results of the queries.
An output of the format
```
RESULT algo=O(1) name=paul_kailer time=xms space=y
```
is printed.
`O(1)` is the name of the (constant time) algorithm. `x` is the amount of time in ms required for all queries, while `y` is the space in bits that were required to allow constant time access to the bitvector.