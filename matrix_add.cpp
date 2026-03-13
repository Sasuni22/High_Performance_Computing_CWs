#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 2048; // Matrix size

int main() {
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C_serial(N, vector<int>(N));
    vector<vector<int>> C_parallel(N, vector<int>(N));

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    // Serial addition
    auto start_serial = high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C_serial[i][j] = A[i][j] + B[i][j];
    auto end_serial = high_resolution_clock::now();
    cout << "Serial addition time: " 
         << chrono::duration_cast<chrono::milliseconds>(end_serial - start_serial).count() 
         << " ms\n";

    // Parallel addition
    auto start_parallel = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C_parallel[i][j] = A[i][j] + B[i][j];
    auto end_parallel = high_resolution_clock::now();
    cout << "Parallel addition time: " 
         << chrono::duration_cast<chrono::milliseconds>(end_parallel - start_parallel).count() 
         << " ms\n";

    // Print small snippet
    cout << "\nSnippet of result matrix (first 5x5 elements):\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            cout << C_parallel[i][j] << " ";
        cout << endl;
    }

    return 0;
}
