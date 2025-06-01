# Chapter 3 exercise
1.
    a. ex1a folder
    b. ex1b folder
    c. Pros and cons discussion:
        - Each thread one row: given the data is row major, then we can access the data of A and C better, whereas matrix B has to jump around each memory block.
        - Each thread one column: the data of B now is fixed given a thread, so one could get the data and store in an array and it would make it less jump around, however now when write to C it has to jump.
2. ex2 folder
3.
    a. Number of threads per block is (((300-1)/16+1),((150-1)/32+1))=(19,5)
    b. Number of threads in the grid is (304,160)
    c. Number of blocks in the grid is (16,32)
    d. Number of threads that execute the code on line 05 is M*N=450000
4.
    a. A[20*400+10]
    b. A[col*500+20]
5.
    a. A[10*500*300+20*300+5]
