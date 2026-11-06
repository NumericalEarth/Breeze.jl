// A KernelAbstractions launch as lower-kernel-to-polygeist leaves it once its indexing is
// delinearized: the gpu_wrapper keeps the 1-D grid of 512 blocks and 1-D block of 1024 threads,
// while the parallel loop runs over the delinearized dimensions, and no
// enzymexla.kernel_thread_indices says which of them were threads.

module {
  // Three dimensions (512 × 8 × 128): with emit_single_alternative=true this compiles with every
  // dimension on the grid and one thread per block
  func.func @three_dims(%out: memref<512x8x128xf32>, %v: f32) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %w = "enzymexla.gpu_wrapper"(%c512, %c1, %c1, %c1024, %c1, %c1) ({
      scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c512, %c8, %c128) step (%c1, %c1, %c1) {
        memref.store %v, %out[%i, %j, %k] : memref<512x8x128xf32>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %w : index
  }

  // Four dimensions (32 × 16 × 8 × 128, the failing Breeze kernel's): with
  // emit_single_alternative=true, four grid dimensions are one too many, "no block size splits
  // this kernel", and the launch becomes the error code 701
  func.func @four_dims(%out: memref<32x16x8x128xf32>, %v: f32) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %w = "enzymexla.gpu_wrapper"(%c512, %c1, %c1, %c1024, %c1, %c1) ({
      scf.parallel (%i, %j, %k, %l) = (%c0, %c0, %c0, %c0) to (%c32, %c16, %c8, %c128) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i, %j, %k, %l] : memref<32x16x8x128xf32>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %w : index
  }
}
