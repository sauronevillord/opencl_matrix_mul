__kernel void matrix_mul(
    __global float* A,
    __global float* B,
    __global float* C,
    int row_lengthA, int row_lengthB
    ){
        int row_num = get_global_id(0); // su quale riga lavora il work-item?
        int col_num = get_global_id(1); // su quale colonna lavora il work-item?

        float res = 0;
        for (int i = 0; i < row_lengthA; i++){
            int posA = row_num * row_lengthA + i;
            int posB = col_num + row_lengthB * i;
            res += A[posA] * B[posB];
        }
        C[row_lengthB * row_num + col_num] = res;
}