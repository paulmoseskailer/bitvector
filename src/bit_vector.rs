use std::time::Instant;
use std::collections::HashMap;
use std::fs::File;

use crate::util::*;

#[path = "./util.rs"]
#[macro_use]
pub mod util;

macro_rules! debug_print {
    ($($expr:tt)*) => { if false { print!($($expr)*); } }
}
macro_rules! benchmark_print {
    ($($expr:tt)*) => { if true { print!($($expr)*); } }
}
static SAVE_EVAL : bool = true;

#[derive(Debug)]
pub enum Request{
  Access{i: u64},
  Rank{b:bool, i: u64},
  Select{b:bool, i: u64},
}

#[derive(Debug)]
pub struct BitVector {
  pub required_space_access : u64,
  pub required_space_rank : u64,
  pub required_space_select : u64,
  pub n : u64,

  // word-size of 64 bits, see lecture slides
  pub data : Vec<u64>,

  // -------------------

  // rank support blocks
  pub s : usize,       // block size
  pub s_prime : usize, // superblock size
  pub num_zeros_superblock : Vec<u64>,
  pub num_zeros_block : Vec<u64>,
  // store the result for every possible query in every possible block
  // of size s.
  // indices: block, index inside block
  pub block_index_ranks : HashMap<(u64, u64), u64>,

  // -------------------

  // select support blocks
  pub k : u64,          // number of zeros in bitvector
  pub b : usize,        // = (lg n)^2, number of zeros per superblock
  pub b_prime : usize,  // = sqrt(lg n), number of zeros per block

  // indices: superblock, b in {0,1}
  pub superblock_ends : HashMap<(u64,bool),u64>,
  // indices: superblock, b in {0,1}
  pub superblock_stored_naively : HashMap<(u64, bool), bool>,
  // indices: superblock, index_inside_superblock, b in {0,1} 
  pub select_inside_superblock : HashMap<(u64, u64, bool), u64>,
  // indices: superblock, block, b in {0,1}
  pub block_ends : HashMap<(u64, u64, bool), u64>,
  // indices: superblock, index, b in {0,1}
  pub block_stored_naively : HashMap<(u64, u64, bool), bool>,
  // indices: superblock, block, index_inside_block, b in {0,1}
  pub select_inside_block : HashMap<(u64, u64, u64, bool), u64>,
  // indices: block, block width, index_inside_block, b in {0,1}
  // assuming n < 2^64, thus block size b_prime < lg(n) = 64,
  // thus passing block as u64 is fine
  pub select_from_block : HashMap<(u64, u64, u64, bool), u64>,
}

// calculates the rank of every possible index in every possible block
fn generate_block_index_ranks(block_size : usize) -> HashMap<(u64, u64), u64> {
  let max_block_value = 2_u64.pow(block_size as u32) - 1;
  let mut block_index_ranks: HashMap<(u64, u64), u64> = HashMap::new();

  for block in 0..(max_block_value+1) {
    let left_shifted_block = block << (63 - (block_size-1));
    for i in 0..block_size {
      let mut num_zeros : u64 = 0;
      // add the value of the previous index
      if i > 0 {
        num_zeros += block_index_ranks.get(&(block, (i-1).try_into().unwrap())).unwrap();
        // index i adds +1 to the rank iff AND with a 1 at the previous index is zero
        num_zeros += (((left_shifted_block) & (1u64 << (63 - i + 1))) == 0) as u64;
      }
      block_index_ranks.insert((block, i.try_into().unwrap()), num_zeros);
    }
  }
  block_index_ranks
}

// calculate select result of every possible index in every possible block
fn generate_block_index_selects(n : u64) -> HashMap<(u64,u64,u64,bool), u64> {
  let mut block_index_selects : HashMap<(u64,u64, u64,bool), u64> = HashMap::new();
  let max_block_size = n.ilog2()/2 as u32;
  debug_print!("MAX_BLOCK_SIZE {max_block_size}\n");
  benchmark_print!("MAX_BLOCK_SIZE {max_block_size}\n");
  for block_width in 0..max_block_size {
    benchmark_print!("Generating block width {block_width}\n");
    let max_block_size_for_width = 2_u64.pow(block_width);
    for block in 0..max_block_size_for_width {
      debug_print!("saving block w={block_width}: {block:b}\n");
      // "0th" one/zero is at the end of the previous block -> store 0
      block_index_selects.insert((block, block_width as u64, 0, false), 0);
      block_index_selects.insert((block, block_width as u64, 0, true), 0);

      let mut index_inside_block = 0;
      let mut zero_counter = 1;
      let mut one_counter = 1;
      while index_inside_block < block_width {
        if ((block >> (block_width - 1 - index_inside_block)) & (1_u64)) == 0 {
          block_index_selects.insert((block, block_width as u64, zero_counter, false), index_inside_block as u64);
          zero_counter += 1;
        } else {
          block_index_selects.insert((block, block_width as u64, one_counter, true), index_inside_block as u64);
          one_counter += 1;
        }
        index_inside_block += 1;
      }
    }
  }
  block_index_selects
}

pub fn bitvector_from_datastring(string:String) -> BitVector {
  let n = string.len() as u64;
  let s = (n.ilog2()/2) as usize;

  let num_words = (n as f64 / 64.0).ceil() as usize;
  let mut a:Vec<u64> = vec![0; num_words];

  debug_print!("have n = {}, num_words = {} \n", n, num_words);

  let chars : Vec<char> = string.chars().collect();
  let mut num_ones = 0;
  for word_index in 0..(num_words - 1) {
    let mut word : u64 = 0;
    for char_index in 0..64 {
      let c = chars[(64*word_index) + char_index];
      if c == '1' {
        word = word | ((1 as u64) << (63 - char_index));
        num_ones += 1;
      }
    }
    a[word_index] = word;
  }
  // deal with left over chars (since n might not be divisible by 64)
  let mut bit_index = 0;
  let mut word : u64 = 0;
  for char_index in ((num_words-1)*64)..(chars.len()) {
    if chars[char_index] == '1' {
      word = word | ((1 as u64) << (63 - bit_index));
      num_ones += 1;
    }
    bit_index += 1;
  }
  a[num_words-1] = word;

  create_bitvector(a, n, s, n-num_ones)
}

pub fn create_bitvector(data: Vec<u64>, n: u64, s: usize, k: u64) -> BitVector {
  let b = ((n as f64).log2() * (n as f64).log2()) as usize;
  let b_prime = (n as f64).log2().sqrt() as usize;

  // In order to use access queries to populate the rank & select support datastructures,
  // a bitvector with only access functionality is instantiated.
  // Yes, this is wasteful, but this project only evaluates the performance of a static bitvector.
  let bv_no_rank_no_select = BitVector {
    data: data.clone(),
    n, s, k, b, b_prime,
    s_prime : s*s,
    num_zeros_superblock : Vec::new(),
    num_zeros_block : Vec::new(),
    block_index_ranks: HashMap::new(),
    superblock_ends : HashMap::new(),
    select_inside_superblock : HashMap::new(),
    superblock_stored_naively : HashMap::new(),
    block_ends : HashMap::new(),
    block_stored_naively : HashMap::new(),
    select_inside_block : HashMap::new(),
    select_from_block : HashMap::new(),
    required_space_access : 0,
    required_space_rank : 0,
    required_space_select : 0
  };

  let rank_start = Instant::now();
  let (num_zeros_superblock, num_zeros_block) = get_rank_support(&bv_no_rank_no_select);
  let block_index_ranks = generate_block_index_ranks(s);
  benchmark_print!("Rank support in {} ms\n", rank_start.elapsed().as_millis());
  
  let sel_start = Instant::now();
  let (superblock_ends, superblock_stored_naively, select_inside_superblock, block_ends, block_stored_naively, select_inside_block) = get_select_support(&bv_no_rank_no_select);
  benchmark_print!("Select support in {} ms\n", sel_start.elapsed().as_millis());
  let block_sel_start = Instant::now();
  let select_from_block = generate_block_index_selects(n);
  benchmark_print!("Select blocks in {} ms\n", block_sel_start.elapsed().as_millis());

  // For printing to the console as required by the project description;
  let required_space_access : usize = (n * 64).try_into().unwrap();
  let required_space_rank = 
        required_space_access
      + num_zeros_block.len() * 64
      + num_zeros_superblock.len() * 64
      + block_index_ranks.allocated_size();
  let required_space_select = 
        required_space_access
      + superblock_ends.allocated_size()
      + superblock_stored_naively.allocated_size()
      + select_inside_superblock.allocated_size()
      + block_ends.allocated_size()
      + block_stored_naively.allocated_size()
      + select_inside_block.allocated_size()
      + select_from_block.allocated_size();

  debug_print!("have k={}, b={}, k/b={} \n", k, b, k/(b as u64));
  BitVector {
    data, n, s, k, b, b_prime, 
    s_prime : s*s,
    num_zeros_superblock,
    num_zeros_block,
    block_index_ranks,
    superblock_ends,
    superblock_stored_naively,
    select_inside_superblock,
    block_ends,
    block_stored_naively,
    select_inside_block,
    select_from_block,
    required_space_access : required_space_access.try_into().unwrap(),
    required_space_rank : required_space_rank.try_into().unwrap(),
    required_space_select : required_space_select.try_into().unwrap(),
  }
}

// Calculate how many zeros from beginning of the bitvector until the end of every 
// superblock (for all superblocks),
// and how many zeros from beginning of the superblock until the end of every block 
// (if superblock is divided into blocks)
fn get_rank_support(bv : &BitVector) -> (Vec<u64>, Vec<u64>) {
  let n = bv.n;
  let s = bv.s; // block size
  let s_prime = bv.s_prime; // superblock size
  let mut num_zeros_block = vec![0; (n/(s as u64)) as usize];
  let mut num_zeros_superblock = vec![0; (n/(s_prime as u64)) as usize];

  // loop over superblocks ...
  let num_superblocks : usize = (n/(s_prime as u64)) as usize;
  for super_block_index in 0..num_superblocks {
    // ... and loop over all s blocks inside the superblock
    for block_index in 0..s {
      let mut this_block_zeros = 0;
      if block_index > 0 {
        // add zeros from previous blocks inside the superblock, see lecture slides
        this_block_zeros = num_zeros_block[super_block_index*s + block_index - 1];
      }
      for bit_inside_block in 0..s {
        let bv_index = super_block_index*s_prime + block_index*s + bit_inside_block;
        if execute_access(bv, &(bv_index as u64)) == 0 {
          this_block_zeros += 1;
        }
      }
      num_zeros_block[super_block_index*s + block_index] = this_block_zeros;
    }
    let mut this_sblock_zeros = 0;
    if super_block_index > 0 {
      // ... add number of zeros from beginning of array to beginning of superblock ...
      this_sblock_zeros += num_zeros_superblock[super_block_index-1];
    }
    // superblock number of zeros is just the sum of zeros in all s blocks
    this_sblock_zeros += num_zeros_block[super_block_index*s + (s-1)];
    num_zeros_superblock[super_block_index] = this_sblock_zeros;
  }
  // add values for blocks after the last superblock
  let num_blocks_after_last_sblock = (n as usize - num_superblocks * s_prime) / s;
  for block in 0..(num_blocks_after_last_sblock as usize) {
    let mut this_block_zeros = 0;
    if block > 0 {
      // add zeros from previous blocks inside the superblock, see lecture slides
      this_block_zeros = num_zeros_block[num_superblocks*s + block - 1];
    }
    for bit_inside_block in 0..s {
      let bv_index = num_superblocks*s_prime + block*s + bit_inside_block;
      if execute_access(bv, &(bv_index as u64)) == 0 {
        this_block_zeros += 1;
      }
    }
    num_zeros_block[num_superblocks*s + block] = this_block_zeros;
  }
  (num_zeros_superblock, num_zeros_block)
}

// Determine sizes of the (super)blocks, store some blocks select query answers naively
fn get_select_support(bv : &BitVector) -> (HashMap<(u64,bool),u64>, HashMap<(u64,bool),bool>, HashMap<(u64,u64,bool),u64>, HashMap<(u64,u64,bool),u64>, HashMap<(u64,u64,bool),bool>, HashMap<(u64,u64,u64,bool),u64>) {
  // Need support structures for both select_0 and select_1
  let (mut superblock_ends_0, mut superblock_stored_naively_0, mut select_inside_superblock_0, mut block_ends_0, mut block_stored_naively_0, mut select_inside_block_0) = get_select_support_b(bv, false);
  let (superblock_ends_1, superblock_stored_naively_1, select_inside_superblock_1, block_ends_1, block_stored_naively_1, select_inside_block_1) = get_select_support_b(bv, true);
  superblock_ends_0.extend(superblock_ends_1);
  superblock_stored_naively_0.extend(superblock_stored_naively_1);
  select_inside_superblock_0.extend(select_inside_superblock_1);
  block_ends_0.extend(block_ends_1);
  block_stored_naively_0.extend(block_stored_naively_1);
  select_inside_block_0.extend(select_inside_block_1);
  (superblock_ends_0, superblock_stored_naively_0, select_inside_superblock_0, block_ends_0, block_stored_naively_0, select_inside_block_0)
}

// Select support structures for either 0 or 1, depending on value_to_select
fn get_select_support_b(bv : &BitVector, value_to_select : bool) -> (HashMap<(u64,bool),u64>, HashMap<(u64,bool),bool>, HashMap<(u64,u64,bool),u64>, HashMap<(u64,u64,bool),u64>, HashMap<(u64,u64,bool),bool>, HashMap<(u64,u64,u64,bool),u64>) {
  let n = bv.n;
  let k = match value_to_select {
    false => bv.k,    // number of zeros
    true => n - bv.k, // number of ones
  };
  let b = bv.b;
  let b_prime = (n.ilog2() as f64).sqrt() as usize;
  let mut num_superblocks = (k/b as u64) as usize;
  // round up (didn't find a good enough built-in function for u64 division)
  if ((num_superblocks*b) as u64) < k {
    num_superblocks += 1;
  }
  let num_blocks_per_superblock = (b as f64/ b_prime as f64).ceil() as usize;
  debug_print!("k/b={}, num_sblocks: {num_superblocks}, num b per sb {num_blocks_per_superblock}\n", k as usize/b );
  debug_print!("b={b}, b_prime={b_prime}\n");

  let mut superblock_ends : HashMap<(u64, bool), u64> = HashMap::new();
  let mut superblock_stored_naively : HashMap<(u64,bool), bool> = HashMap::new();
  let mut select_inside_superblock : HashMap<(u64,u64,bool),u64> = HashMap::new();
  let mut block_ends : HashMap<(u64,u64,bool),u64> = HashMap::new();
  let mut block_stored_naively : HashMap<(u64,u64,bool), bool> = HashMap::new();
  let mut select_inside_block : HashMap<(u64,u64,u64,bool),u64> = HashMap::new();

  // iterate over bitvector, saving the end of each superblock
  let mut bv_index : u64 = 0;
  for superblock in 0..num_superblocks {
    let mut zeros_left = b;
    while zeros_left > 0 && bv_index < n{
      if execute_access(&bv, &bv_index) == value_to_select as u64 {
        zeros_left -= 1;
      }
      bv_index += 1;
    }
    // superblock_end is start of next superblock
    let superblock_end = bv_index;
    superblock_ends.insert((superblock as u64, value_to_select), superblock_end);
    let superblock_start = match superblock > 0 {
      true => superblock_ends.get(&((superblock-1) as u64, value_to_select)).unwrap().clone(),
      false => 0,
    };
    let superblock_size = superblock_end - superblock_start;
    // superblock spans (superblock_start) until including (superblock_end - 1)

    // save results for select inside the superblock: select_0(B_{i/b}, i - (i/b * b))
    // for explanation on which size of superblock is stored naively see lecture slide 10
    if superblock_size >= n.ilog2().pow(4) as u64 {
      debug_print!("SAVING sblock {superblock} naively!\n");
      superblock_stored_naively.insert((superblock as u64, value_to_select), true);
      // store answers naively
      // iterate through superblock, save position of every 0
      let mut inside_sblock_index = 0;
      for request_index in 1..(b as u64) {
        while execute_access(&bv, &(superblock_start + inside_sblock_index)) != value_to_select as u64 {
          inside_sblock_index += 1;
        }
        select_inside_superblock.insert(((superblock as u64), request_index, value_to_select), inside_sblock_index);
      }
      // the "0th" zero is the end of the last superblock
      select_inside_superblock.insert((superblock as u64, 0, value_to_select), 0);
    } else {
      debug_print!("NOT SAVING sblock {superblock} naively\n");
      superblock_stored_naively.insert((superblock as u64, value_to_select), false);
      // divide superblock into blocks of variable size, with b_prime zeros each

      // copy block ends to seperate vector for convenience 
      // (don't care about performance here, only the static bv is evaluated)
      let mut tmp_block_ends = vec![0; num_blocks_per_superblock as usize];

      let mut bv_index = superblock_start;
      for block in 0..num_blocks_per_superblock {
        // same iterating approach as with superblocks above:
        // determine the end of every block by counting zeros
        let mut zeros_left = b_prime;
        while zeros_left > 0 && bv_index < n{
          if execute_access(&bv, &bv_index) == value_to_select as u64 {
            zeros_left -= 1;
          }
          bv_index += 1;
        }
        let block_end = bv_index;
        tmp_block_ends[block as usize] = block_end;
        block_ends.insert((superblock as u64,block as u64, value_to_select), block_end);
        
        let block_start = match block > 0 {
          true => tmp_block_ends[(block-1) as usize],
          false => 0,
        };
        let block_size = block_end - block_start;

        // the suggestion on lecture slide 10 of lg n does not work well, (lg n)/2 is better
        let this_block_stored_naively : bool = block_size >= (n.ilog2()/2) as u64;
        block_stored_naively.insert((superblock as u64, block as u64, value_to_select), this_block_stored_naively);

        if this_block_stored_naively {
          debug_print!("SAVING for b={value_to_select} sblock {superblock}, block {block}, i in {block_start}..{block_end} naively\n");
          // store all answers naively
          // iterate through block, save position of every 0
          let mut inside_block_index = 0;
          for request_index in 1..(b_prime as u64) {
            while execute_access(&bv, &(block_start + inside_block_index)) != value_to_select as u64 {
              inside_block_index += 1;
            }
            assert!(inside_block_index < block_size, "inside_block_index increased too far, should have hit a value_to_select!");
            select_inside_block.insert(((superblock as u64), block as u64, request_index, value_to_select), inside_block_index);
            inside_block_index += 1;
          }
        } else {
          debug_print!("SAVING for b={value_to_select} sblock {superblock}, block {block}, i in {block_start}..{block_end} by lookup\n");
          // select inside this block is calculated by lookup of the entire block in select_from_block
        } 
      }
    }
  }

  (superblock_ends, superblock_stored_naively, select_inside_superblock, block_ends, block_stored_naively, select_inside_block)
}

pub fn handle_request(a: &BitVector, request: &Request) -> (u64, u128) {
  benchmark_print_request(request);
  let start = Instant::now();
  let result = match request {
    Request::Access {i} => execute_access(a, i),
    Request::Rank {b, i}=> execute_rank(a, b, i),
    Request::Select {b, i}=> execute_select(a, b, *i),
  };
  let dt = start.elapsed();

  let space_required = match request {
    Request::Access {i : _} => a.required_space_access,
    Request::Rank {b : _, i : _}=> a.required_space_rank,
    Request::Select {b : _, i : _}=> a.required_space_select,
  };
  benchmark_print!("{}, request took {} ms \n", result, dt.as_micros() / 1000);
  println!("RESULT algo=O(1) name=paul_kailer time={} space={}", dt.as_micros() / 1000, space_required);
  (result, dt.as_micros()/1000)
}

fn benchmark_print_request(request: &Request) {
  match request {
    Request::Access {i} => benchmark_print!("access({}) = ", i),
    Request::Rank {b, i}=> benchmark_print!("rank_{} ({}) = ", b, i),
    Request::Select {b, i}=> benchmark_print!("select_{} ({}) = ", b, i),
  };
}

fn execute_access(a:&BitVector, index : &u64) -> u64 {
  // see lecture slides
  let block : u64 = a.data[(index/64) as usize];
  (block >> ((63 - (index.rem_euclid(64))))) & 1u64
}

fn execute_rank(a:&BitVector, b: &bool, index : &u64) -> u64 {
  assert!(!a.num_zeros_block.is_empty(), "Rank query needs instantiated num_zeros_block");
  assert!(!a.num_zeros_superblock.is_empty(), "Rank query needs instantiated num_zeros_superblock");
  if *b {
    // rank_1(i) = i - rank_0(i)
    return (*index as u64) - execute_rank(a, &false, index)
  }

  let sblock_index = index / (a.s_prime as u64);
  let block_index = (index - (a.s_prime as u64)*sblock_index) / (a.s as u64);
  let mut zeros_to_block = 0;
  if sblock_index > 0 {
    zeros_to_block += a.num_zeros_superblock[(sblock_index - 1) as usize];
  }
  if block_index > 0 {
    zeros_to_block += a.num_zeros_block[(a.s * sblock_index as usize) + (block_index - 1) as usize];
  }
  zeros_to_block + get_rank_inside_block(a, index)
}

fn get_rank_inside_block(bv : &BitVector, index: &u64) -> u64 {
  let index_in_block = *index % (bv.s as u64);
  let block_start = *index - index_in_block;

  // get block via s = lg(n)/2 access queries
  let block : u64 = get_bits(bv, block_start, block_start + (bv.s as u64));

  *bv.block_index_ranks.get(&(block, index_in_block)).unwrap()
}

fn execute_select(bv:&BitVector, b: &bool, index : u64) -> u64 {
  debug_print!("SELECT_START\n gettig prefix sum for i={}\n", index);
  if *b {
    assert!(index <= (bv.n - bv.k), "Invalid select index requested!");
  } else {
    assert!(index <= bv.k, "Invalid select index requested!");
  }
  
  let superblock_index = (index/(bv.b as u64)) as usize;
  let index_inside_superblock = index - (superblock_index * bv.b) as u64;
  let block_index = index_inside_superblock / bv.b_prime as u64;
  let index_inside_block = index_inside_superblock - (block_index * bv.b_prime as u64);
  assert!(index_inside_block < bv.b_prime as u64);
  debug_print!("for b={b}, i={index}, sblock i={superblock_index}, inside sblock i={index_inside_superblock}, block={block_index} inside b i={index_inside_block}\n");

  let prefix_sum = match superblock_index {
    0 => 0,
    _ => bv.superblock_ends.get(&((superblock_index - 1) as u64, *b)).unwrap().clone() - 1,
  };
  debug_print!("prefix sum={}\n", prefix_sum);
  if index_inside_superblock == 0 {
    debug_print!("SELECT END, index inside sblock is 0!\n");
    return prefix_sum;
  }
  if index_inside_block == 0 {
    debug_print!("SELECT END, index inside block is 0!\n");
    return match block_index {
      0 => 0,
      _ => bv.block_ends.get(&(superblock_index as u64, block_index-1, *b)).unwrap().clone() - 1,
    }
  }

  let mut select_inside_sblock;
  if *bv.superblock_stored_naively.get(&(superblock_index as u64, *b)).unwrap() {
    debug_print!("superblock stored naively!\n");
    select_inside_sblock = bv.select_inside_superblock.get(&(superblock_index as u64, index_inside_superblock, *b)).unwrap().clone();
  } else {
    select_inside_sblock = match block_index {
      0 => 0,
      _ => bv.block_ends.get(&(superblock_index as u64, block_index - 1, *b)).unwrap().clone() - prefix_sum,
    };
    debug_print!("superblock not stored naively, block starts at {select_inside_sblock} \n");
    if *bv.block_stored_naively.get(&(superblock_index as u64, block_index, *b)).unwrap() {
      debug_print!("block stored naively!\n");
      select_inside_sblock += bv.select_inside_block.get(&(superblock_index as u64, block_index, index_inside_block, *b)).unwrap().clone();
    } else {
      select_inside_sblock += get_select_from_block(&bv, superblock_index, block_index, index_inside_block, *b);
      debug_print!("block lookup result={}\n",get_select_from_block(&bv, superblock_index, block_index, index_inside_block, *b));
    }
  };
  debug_print!("SELECT END, result = {prefix_sum} + {select_inside_sblock}\n");
  prefix_sum + select_inside_sblock
}

// if block is not stored naively, have to lookup the entire block (and requested index) in select_from_block HashMap
fn get_select_from_block(bv: &BitVector, superblock_index : usize, block_index : u64, index_inside_block : u64, b : bool) -> u64 {
  // getting the block takes block_size * access_request = lg(n) * O(1) = lg(n) time
  let block_start = match block_index > 0 {
    true => bv.block_ends.get(&(superblock_index as u64, block_index-1, b)).unwrap().clone(),
    false => match superblock_index > 0 {
      true => bv.superblock_ends.get(&((superblock_index-1) as u64, b)).unwrap().clone(),
      false => 0,
    }
  };
  let block_end = bv.block_ends.get(&(superblock_index as u64, block_index, b)).unwrap().clone();
  let block_width = block_end - block_start;
  let block = get_bits(bv, block_start, block_end);
  debug_print!("Select block lookup b={b}: width={block_width}, block={block:064b}, i={index_inside_block}\n");
  bv.select_from_block.get(&(block, block_width, index_inside_block, b)).unwrap().clone()
}

fn get_bits(bv:&BitVector, start : u64, end : u64) -> u64 {
  let mut bits : u64 = 0;
  let width = end - start;
  assert!(width <= 64, "cannot get more than 64 bits simultaneously");

  for i in start..end {
    let distance_from_start = i - start;
    if execute_access(bv, &i) == 1 {
      bits = bits | (1_u64 << (width - 1 - distance_from_start));
    }
  }
  bits
}


#[cfg(test)]
mod tests {
  use super::*;

  fn get_test_vector() -> BitVector {
    let test_string : String= String::from("1010011101001001");
    let result = bitvector_from_datastring(test_string);
    result
  }
  fn get_long_test_vector() -> BitVector {
    let test_string_80 : String= String::from("10100111010010011010011101001001101001110100100110100111010010011010011101001001");
    let result = bitvector_from_datastring(test_string_80);
    println!("bv 80: {:?}", result);
    result
  }
  fn get_eighty(number: bool) -> BitVector {
    let test_string_80 : String= match number {
      true => String::from("11111111111111111111111111111111111111111111111111111111111111111111111111111111"),
      false => String::from("00000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    };
    let result = bitvector_from_datastring(test_string_80);
    println!("bv 80: {:?}", result);
    result
  }

  #[test]
  fn test_eighty_ones() {
    let a : BitVector = get_eighty(true);
    print!("Input read: n_vector={} a=", a.n);
    for x in a.data.iter() {
      print!("{x:064b}");
    }
    println!("");
    assert_eq!(execute_access(&a, &0), 1);
    assert_eq!(execute_access(&a, &63), 1);
    assert_eq!(execute_access(&a, &79), 1);
    assert_eq!(execute_rank(&a, &false, &0), 0);
    assert_eq!(execute_rank(&a, &false, &1), 0);
    assert_eq!(execute_rank(&a, &false, &36), 0);
    assert_eq!(execute_rank(&a, &false, &79), 0);
    assert_eq!(execute_select(&a, &true, 1), 0);
    assert_eq!(execute_select(&a, &true, 80), 79);
  }

  #[test]
  fn test_eighty_zeros() {
    let a : BitVector = get_eighty(false);
    print!("Input read: n_vector={} a=", a.n);
    for x in a.data.iter() {
      print!("{x:064b}");
    }
    println!();
    assert_eq!(execute_access(&a, &0), 0);
    assert_eq!(execute_access(&a, &63), 0);
    assert_eq!(execute_access(&a, &79), 0);
    assert_eq!(execute_rank(&a, &false, &0), 0);
    assert_eq!(execute_rank(&a, &false, &1), 1);
    assert_eq!(execute_rank(&a, &false, &74), 74);
    assert_eq!(execute_rank(&a, &false, &75), 75);
    assert_eq!(execute_rank(&a, &false, &76), 76);
    assert_eq!(execute_rank(&a, &false, &79), 79);
    assert_eq!(execute_select(&a, &false, 1), 0);
    assert_eq!(execute_select(&a, &false, 80), 79);
  }

  #[test]
  fn test_access() {
    let a : BitVector = get_test_vector();
    assert_eq!(execute_access(&a, &0), 1);
    assert_eq!(execute_access(&a, &3), 0);
    assert_eq!(execute_access(&a, &7), 1);
  }

  #[test]
  fn test_rank() {
    let a : BitVector = get_test_vector();
    assert_eq!(execute_rank(&a, &false, &0), 0);
    assert_eq!(execute_rank(&a, &false, &6), 3);
    let a : BitVector = get_long_test_vector();
    assert_eq!(execute_rank(&a, &false, &0), 0);
    assert_eq!(execute_rank(&a, &false, &1), 0);
    assert_eq!(execute_rank(&a, &true, &36), 18);
    assert_eq!(execute_rank(&a, &true, &79), 39);
    assert_eq!(execute_rank(&a, &true, &80), 40);
  }

  #[test]
  fn test_select() {
    let a : BitVector = get_test_vector();
    println!("bv:");
    println!("{a:?}");
    assert_eq!(execute_select(&a, &false, 1), 1);
    assert_eq!(execute_select(&a, &true, 1), 0);
    assert_eq!(execute_select(&a, &true, 5), 7);
    let a : BitVector = get_long_test_vector();
    assert_eq!(execute_select(&a, &true, 1), 0);
    assert_eq!(execute_select(&a, &false, 1), 1);
    assert_eq!(execute_select(&a, &true, 3), 5);
    assert_eq!(execute_select(&a, &false, 4), 8);
  }
}