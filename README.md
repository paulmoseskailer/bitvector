# Advanced Data Structures Project

This repo contains my solution to the project alongisde the SS24 Advanced Data Structures lecture.

It implements a bit vector that supports

1. access
2. rank
3. select

## Input Output

Invocation via 

`executable input_file output_file`

Input is an integer $n \in \mathbb{N}$ followed by the bitvector $a \in 2^{\{0,1\}}$.
$n$ subsequent lines contain one access, rank or select request each.

## Benchmark

To demonstrate efficiency of the data structure, I track the performance of

`cargo run -- input/input_2to27.txt`

through different implementations.

### `Vec<bool>`

Implemented as a naive vector of bools at [this commit](43ced597af43ab76a210e371fef0ed39ee8cd659).

```
Input processed in 2031 ms
access(0) = 1, request took 0 ms
access(1591298) = 0, request took 0 ms
access(28340101) = 1, request took 0 ms
access(104217728) = 1, request took 0 ms
access(133167421) = 0, request took 0 ms
rank_true (100002) = 50173, request took 2 ms
rank_true (12000101) = 6048267, request took 329 ms
rank_true (99999999) = 50394371, request took 2735 ms
rank_true (104217727) = 52520146, request took 3043 ms
rank_true (133167421) = 67108402, request took 3767 ms
select_false (3001) = 5948, request took 0 ms
select_false (770000) = 1551486, request took 31 ms
select_false (8800901) = 17741219, request took 322 ms
select_false (20200021) = 40715986, request took 664 ms
select_false (54217728) = 109298034, request took 1785 ms
```

### `Vec<u64>`, rank support with (super)blocks, select in O(n) time

Access and rank implemented as in the lecture slides.
Select in primitive implementation, i.e. O(n) time and no extra space.

```
Input processed in 8211 ms
access(0) = 1, request took 0 ms
access(1591298) = 0, request took 0 ms
access(28340101) = 1, request took 0 ms
access(104217728) = 1, request took 0 ms
access(133167421) = 0, request took 0 ms
rank_true (100002) = 50173, request took 0 ms
rank_true (12000101) = 6048268, request took 0 ms
rank_true (99999999) = 50394371, request took 0 ms
rank_true (104217727) = 52520147, request took 0 ms
rank_true (133167421) = 67108402, request took 0 ms
select_false (3001) = 5948, request took 0 ms
select_false (770000) = 1551486, request took 34 ms 
select_false (8800901) = 17741219, request took 401 ms 
select_false (20200021) = 40715986, request took 854 ms 
select_false (54217728) = 109298034, request took 2342 ms 
```

Evidently great improvement for rank queries.
Note how select queries take comparably long if not even a bit worse.

### `Vec<u64>`, rank support with (super)blocks, select with (super)blocks

At [this commit](2f6ce1c4d91a0d17f5aeb5fae0c33af24b88fc68).

```
access(0) = 1, request took 0 ms 
access(1591298) = 0, request took 0 ms
access(28340101) = 1, request took 0 ms
access(104217728) = 1, request took 0 ms
access(133167421) = 0, request took 0 ms
rank_true (100002) = 50173, request took 1 ms
rank_true (12000101) = 6048268, request took 1 ms
rank_true (99999999) = 50394371, request took 1 ms
rank_true (104217727) = 52520147, request took 1 ms
rank_true (133167421) = 67108402, request took 1 ms
select_false (3001) = 5948, request took 3 ms 
select_false (770000) = 1551486, request took 2 ms
select_false (8800901) = 17741219, request took 4 ms
select_false (20200021) = 40715986, request took 3 ms
select_false (54217728) = 109298034, request took 2 ms
```

Select queries are significantly faster now, too.