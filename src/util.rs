use std::collections::HashMap;

// Adapted from https://stackoverflow.com/questions/68255026/how-to-get-memory-occupied-by-an-object-in-rust-from-the-code-itself

pub trait AllocatedSize {
  fn allocated_size(&self) -> usize;
}

impl AllocatedSize for u64 {
  fn allocated_size(&self) -> usize {
    64
  }
}

impl AllocatedSize for bool {
  fn allocated_size(&self) -> usize {
    8 // Assuming no compiler optimization
  }
}

impl<A: AllocatedSize, B: AllocatedSize> AllocatedSize for (A,B) {
  fn allocated_size(&self) -> usize {
    self.0.allocated_size() + self.1.allocated_size()
  }
}
impl<A: AllocatedSize, B: AllocatedSize, C: AllocatedSize> AllocatedSize for (A,B,C) {
  fn allocated_size(&self) -> usize {
    self.0.allocated_size() + self.1.allocated_size() + self.2.allocated_size()
  }
}
impl<A: AllocatedSize, B: AllocatedSize, C: AllocatedSize, D: AllocatedSize> AllocatedSize for (A,B,C,D) {
  fn allocated_size(&self) -> usize {
    self.0.allocated_size() + self.1.allocated_size() + self.2.allocated_size() + self.3.allocated_size()
  }
}

impl<K: AllocatedSize, V: AllocatedSize> AllocatedSize for HashMap<K, V> {
    fn allocated_size(&self) -> usize {
        // every element in the map directly owns its key and its value
        let element_size: usize = std::mem::size_of::<K>() + std::mem::size_of::<V>();

        // directly owned allocation
        // NB: self.capacity() may be an underestimate, see its docs
        // NB: also ignores control bytes, see hashbrown implementation
        let directly_owned = self.capacity() * element_size;

        // transitively owned allocations
        let transitively_owned : usize = self
            .iter()
            .map(|(key, val)| key.allocated_size() + val.allocated_size())
            .sum();

        directly_owned + transitively_owned
    }
}