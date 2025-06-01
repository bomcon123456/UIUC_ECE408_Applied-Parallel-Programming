# Chapter 4 exercise
1.  a. Number of warps per block: 128 / 32 = 4
    b. Number of warps in grid: 4 * 8 = 32
    c.
        i. <40 needs two warps. >=104 needs another (for the first block). All other blocks activate all warps. So 3 + 8 = 24. (Correction: All warps are active anyway, so the answer should be 32).
        ii. 2 per block, so a total of 16.
        iii. 100%
        iv. 8/32 = 1/4
        v. 3/4
    d.
        i. 32
        ii. 32
        iii. 50%
    e.
        i. 3 because max(i%3) = 2 when i=0, hence the loop condition will diverge when j >= 5-2=3.
        ii. The rest (5-3=2).

2.  ceil(2000/512) * 512

3.  The last two warps.

4.  Total wait time is 4.1, so the percentage is around 17%.

5.  No, even if there are only 32 threads, one should still use __syncthreads() when needed. This is because there are cases where threads should be at the same spot before executing the next one to have correct output. Although currently a warp has 32 threads (which still can be changed in the future), it doesn't guarantee that all threads within that warp will be at the same spot at any given time (this might be due to divergence).

6.  b

7.  a. 50%
    b. 50%
    c. 50%
    d. 100%
    e. 100%

8.  a. Can only use 50% of blocks since 128 * 16 = 2048 (threads) is already the maximum. However, thread occupancy is still maximized at 100% since it can spawn 16 blocks. Also, it uses only 30/32 registers.
    b. Block maximum since the total number of blocks is 64. However, thread occupancy is only 50% since only 32 * 32 = 1024 threads are spawned. Registers also use less, 29/32.
    c. Initially 8 blocks. However, each thread uses more than needed (2), so 1 block will be dropped. Hence, the total threads per SM is 256 * 7 = 1792, and total registers used is 60928/65536.

9.  This would fail since the student spawned too many threads (1024), which is bigger than the device allows (512). Hence, the output is non-determined.
