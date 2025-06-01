1. 
  a. number of warps in per block is 128/32=4
  b. number of waprs in grid is 4*8=32
  c.
    i. <40 needs two warps, >=104 needs another (for the first block),  all other blocks active all warps. so 3+8=24. WRONG!!! all warps are active anyway... so the answer should be 32
    ii. 2 per block so  total of 16
    iii. 100%
    iv. 8/32=1/4
    v. 3/4
  d.
    i. 32
    ii. 32
    iii. 50%
  e.
    i. 3  because max(i%3)==2 when i=0, hence the loop condition will be diverge when j >= 5-2=3
    ii. the rest (5-3=2)
2.  ceil(2000/512)*512
3. the last two warps
4. total wait time is 4.1 so percentage is around 17%
5. no even if therere only 32 threads, one should still sync thread when needed since there are cases where thread should be at the same spot before executing the next one to have correct output. Although right now a warp has 32 threads (which still can be changed in the future), it doesnt guarantee that all thread within that warp will be at the same spot at any given time (might due to divergence).
6. b
7. 
  a. 50%
  b. 50%
  c. 50%
  d. 100%
  e. 100%
8.
  a. can only use 50% block since 128*16=2048 already,  however thread occupancy is still maximize 100% since can spawn 16 block. also, use only 30/32 registers
  b. block max since total of block is 64. however thread occupancy only 50% since only 32*32=1024 threads is spawn, registers  also  use less 29/32
  c. initially 8 blocks, however each thread use more than needed 2, so 1 block will be dropped, hence total thread per sm is 256*7=1792, and total registers used is 60928/65536
9. this would be fail since the student spawn too many threads (1024) bigger than the device allow (512) hence the output is nondetermined.
