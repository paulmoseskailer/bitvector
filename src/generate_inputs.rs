use std::path::Path;
use std::fs::File;
use std::io::Write;
use rand::prelude::*; // TODO remove for final hand-in! (onyl std allwoed)

pub fn generate_inputs(size: u32, num_queries: u32) {
  println!("generating input of size 2^{size}");
  let mut output_file = match File::create(Path::new("inputs/random_input.txt")) {
    Err(why) => panic!("couldn't create output_file: {}", why),
    Ok(file) => file,
  };
  if let Err(e) = writeln!(output_file, "{num_queries}") {
    eprintln!("Couldn't write to file: {}", e);
  }

  let input_length : u64 = 2_u64.pow(size);
  let num_amount = 2_u32.pow(size - 7); //generating 128 = 2^7 bit numbers
  let mut zeros_amount : u64 = 0;
  for _ in 0..num_amount {
    let mut rng = rand::thread_rng();
    let x : u128 = rng.gen_range(0..u128::MAX);
    zeros_amount += x.count_zeros() as u64;
    if let Err(e) = write!(output_file, "{x:0128b}") {
      eprintln!("Couldn't write to file: {}", e);
    }
  }
  if let Err(e) = writeln!(output_file, "") {
    eprintln!("Couldn't write to file: {}", e);
  }

  for _ in 0..num_queries {
    let mut rng = rand::thread_rng();
    let x : u32 = rng.gen();
    let query = match x % 3 {
      0 => generate_random_access(input_length),
      1 => generate_random_rank(input_length),
      2 => generate_random_select(input_length, zeros_amount),
      _ => panic!("modulo went wrong!"),
    };
    if let Err(e) = writeln!(output_file, "{}", query) {
      eprintln!("Couldn't write to file: {}", e);
    }
  }
}

fn generate_random_access(input_length : u64) -> String {
  let mut rng = rand::thread_rng();
  let i = rng.gen_range(0..input_length).to_string();
  let mut query : String = String::from("access ");
  query.push_str(&i);
  query
}
fn generate_random_rank(input_length : u64) -> String {
  let mut rng = rand::thread_rng();
  let i : u64 = rng.gen_range(0..input_length);
  let rand : u32 = rng.gen_range(0..u32::MAX);
  let b : String = match (rand % 2) > 0 {
    true => String::from("0 "),
    false => String::from("1 "),
  };
  let i_str = i.to_string();
  let mut query : String = String::from("rank ");
  query.push_str(&b);
  query.push_str(&i_str);
  query
}
fn generate_random_select(input_length : u64, zeros_amount : u64) -> String {
  let mut rng = rand::thread_rng();
  let rand : u32 = rng.gen_range(0..u32::MAX);
  let b : String = match (rand % 2) > 0 {
    true => String::from("0 "),
    false => String::from("1 "),
  };
  let max_index = match (rand % 2) > 0 {
    true => zeros_amount,
    false => input_length - zeros_amount,
  };
  let i_str : String = rng.gen_range(0..max_index).to_string();
  let mut query : String = String::from("select ");
  query.push_str(&b);
  query.push_str(&i_str);
  query
}