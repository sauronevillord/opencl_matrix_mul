#include <CL/cl.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(){

    const char *kernel_source = "\n" \
    "__kernel void matrix_mul(\n"\
    "__global float* A,\n"\
    "__global float* B,\n"\
    "__global float* C,\n"\
    "int row_lengthA, int row_lengthB\n"\
    "){\n"\
        "int row_num = get_global_id(0);\n"\
        "int col_num = get_global_id(1);\n"\
        "float res = 0;\n"\
        "for (int i = 0; i < row_lengthA; i++){\n"\
            "int posA = row_num * row_lengthA + i;\n"\
            "int posB = col_num + row_lengthB * i;\n"\
            "res += A[posA] * B[posB];\n"\
        "}\n"\
        "C[row_lengthB * row_num + col_num] = res;\n"\
"}\n"\
"\n";


    int a_rows, a_cols;
    int b_rows, b_cols;

    cout << "Digita il numero di righe della prima matrice" << endl;
    cin >> a_rows;
    
    cout << "Digita il numero di colonne della prima matrice" << endl;
    cin >> a_cols;

    b_rows = a_cols;

    cout << "Digita il numero di colonne della seconda matrice" << endl;
    cin >> b_cols;

    cout << "A: " << a_rows << "x" << a_cols << "  B: " << b_rows << "x" << b_cols << endl;

    int sizeA = a_rows * a_cols;
    int sizeB = b_rows * b_cols;
    int sizeC = a_rows * b_cols;

    srand((unsigned int) time(NULL));
    float RAND_RANGE = 30.0;

    size_t a_size = sizeof(float) * sizeA;
    size_t b_size = sizeof(float) * sizeB;
    size_t c_size = sizeof(float) * sizeC;

    float *A = (float*) malloc(a_size);
    float *B = (float*) malloc(b_size);
    float *C = (float*) malloc(c_size);

    for(int i = 0; i < sizeA; i++){
        float el = (float(rand())/float((RAND_MAX)) * RAND_RANGE);
        A[i] = el;
    }
    for(int i = 0; i < sizeB; i++){
        float el = (float(rand())/float((RAND_MAX)) * RAND_RANGE);
        B[i] = el;
    }

    if(sizeA <= 50){
        cout << "A: " << endl;
        for(int i = 0; i < sizeA; i++){
            if(i % a_cols == 0){
                cout << endl;
            }
            cout << A[i] << "   ";
        }
        cout << endl;
    }

    if(sizeB <= 50){
        cout << "B: " << endl;
        for(int i = 0; i < sizeB; i++){
            if(i % b_cols == 0){
                cout << endl;
            }
            cout << B[i] << "   ";
        }
        cout << endl;
    }

    /*
    OPENCL
    */

   cl_context context = 0;
   cl_command_queue command_queue = 0;
   cl_program program = 0;
   cl_device_id device = 0;
   cl_kernel kernel = 0;
   cl_int errNum;

   cl_mem a_mem, b_mem, c_mem;

   cl_uint num_platforms;
   cl_platform_id platform;

   errNum = clGetPlatformIDs(1, &platform, &num_platforms);
   if(errNum != CL_SUCCESS){
       cerr << "Errore nel prendere la piattaforma OpenCL";
   }

   errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(errNum != CL_SUCCESS){
       cerr << "Errore nel prendere il device OpenCL";
   }

    char* value;
    size_t valueSize;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);
    free(value);

   cl_context_properties properties [] = {
       CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0
   };

   context = clCreateContext(properties, 1, &device, NULL, NULL, &errNum);
   if(errNum != CL_SUCCESS){
       cerr << "Errore nel context OpenCL";
   }

   command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errNum);
   if(errNum != CL_SUCCESS){
       cerr << "Errore nella creazione della command queue OpenCL";
   }

   a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, a_size, NULL, NULL);
   b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, b_size, NULL, NULL);
   c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_size, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &errNum);
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(errNum != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        cout << "Errore nella build dell'eseguibile OpenCL!" << endl;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        cerr << buffer;
    }

   kernel = clCreateKernel(program, "matrix_mul", &errNum);

   errNum = 0;
   errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
   errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
   errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
   errNum |= clSetKernelArg(kernel, 3, sizeof(int), &a_cols);
   errNum |= clSetKernelArg(kernel, 4, sizeof(int), &b_cols);

   errNum = clEnqueueWriteBuffer(command_queue, a_mem, CL_TRUE, 0, a_size, A, 0, NULL, NULL);
   errNum = clEnqueueWriteBuffer(command_queue, b_mem, CL_TRUE, 0, b_size, B, 0, NULL, NULL);

   cl_event prof_event;

   size_t global[2] = {(size_t) a_rows, (size_t) b_cols};

   errNum = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, &prof_event);
   clFinish(command_queue);

   cl_ulong ev_start_time = (cl_ulong) 0;
   cl_ulong ev_end_time = (cl_ulong) 0;

   errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
   errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

   errNum = clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0, c_size, C, 0, NULL, NULL);

   cl_ulong run_time = (ev_end_time - ev_start_time) / (1000000000);

   if(a_rows * b_cols <= 50){
       cout << "C: " << endl;
    for(int i = 0; i < sizeC; i++){
            if(i % b_cols == 0){
                cout << endl;
            }
            cout << C[i] << "   ";
        }
        cout << endl;
   }

   cout << "Tempo impiegato: " << run_time << "s " << endl;

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseMemObject(a_mem);
   clReleaseMemObject(b_mem);
   clReleaseMemObject(c_mem);
   clReleaseCommandQueue(command_queue);
   clReleaseContext(context);
   clReleaseDevice(device);

   free(A);
   free(B);
   free(C);
    
}