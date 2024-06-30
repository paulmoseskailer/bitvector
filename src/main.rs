use std::fs::File;
use std::path::Path;
use std::io::{self, prelude::*, BufReader};
use std::env;
use std::time::Instant;

use crate::bit_vector::*;
#[path = "./bit_vector.rs"]
#[macro_use]
pub mod bit_vector;

use crate::generate_inputs::*;
#[path = "./generate_inputs.rs"]
#[macro_use]
pub mod generate_inputs;

static PRINT_INPUT : bool = false; // careful, some input is very large

fn read_input_file(file_path: &str) -> io::Result<(u64, String, Vec<bit_vector::Request>)> {
  let file = File::open(file_path)?;
  let reader = BufReader::new(file);
  let mut lines_iter = reader.lines();
  let n = lines_iter.next().unwrap()?.parse::<u64>().unwrap();
  let bit_string = lines_iter.next().unwrap()?;

  let mut requests:Vec<bit_vector::Request> = Vec::with_capacity(n.try_into().unwrap());
  for line in lines_iter {
    requests.push(parse_request(line?))
  }

  Ok((n, bit_string, requests))
}

fn parse_request(line:String) -> bit_vector::Request {
  let mut parts = line.split_whitespace();
  parts.next(); //skip request word
  if line.contains("access") {
    let i = parts.next().expect("No index given in access request!").parse::<u64>().unwrap();
    bit_vector::Request::Access{i:i}
  } else if line.contains("rank") {
    let b:bool = parts.next().expect("No b given in rank request!").parse::<u64>().unwrap() > 0;
    let i = parts.next().expect("No index given in rank request!").parse::<u64>().unwrap();
    bit_vector::Request::Rank{b:b, i:i}
  } else if line.contains("select") {
    let b:bool = parts.next().expect("No b given in select request!").parse::<u64>().unwrap() > 0;
    let i = parts.next().expect("No index given in select request!").parse::<u64>().unwrap();
    bit_vector::Request::Select{b:b, i:i}
  } else {
    panic!("An input line does not contain any valid request!");
  }
}

fn main() -> io::Result<()> {
  // read input
  let args: Vec<String> = env::args().collect();
  if args.len() < 2 {
    println!("No input file given, generating input...");
    generate_inputs(26, 100);
    return Ok(());
  }

  let output_filepath = match args.len() >= 3 {
    true => &args[2],
    false => "outputs/output.txt"
  };
  let input_filepath = &args[1];
  let input_read_start = Instant::now();
  let (n_queries, data, requests) = read_input_file(input_filepath)?;
  benchmark_print!("Input read, creating bit vector ... \n");
  let vector = bitvector_from_datastring(data);
  assert_eq!(n_queries, requests.len().try_into().unwrap(), "was expecting {} requests, but have {}", n_queries, requests.len());
  benchmark_print!("Input processed in {} ms\n", input_read_start.elapsed().as_micros() / 1000);

  if PRINT_INPUT {
    print!("Input read: n_vector={} a=", vector.n);
    for x in vector.data.iter() {
      print!("{x:064b}");
    }
    println!();
    println!("Vector print: {:?}", vector);
  }
  
  let mut output_file = match File::create(Path::new(output_filepath)) {
    Err(why) => panic!("couldn't create output_file: {}", why),
    Ok(file) => file,
  };

  for request in requests.iter() {
    let result = bit_vector::handle_request(&vector, request);
    // append result to output file
    if let Err(e) = writeln!(output_file, "{}", result) {
      eprintln!("Couldn't write to file: {}", e);
    }
  }
  debug_print!("All requests processed! \n");
  Ok(())
}
