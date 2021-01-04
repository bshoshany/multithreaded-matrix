/**
 * @file matrix_test.cpp
 * @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
 * @version 1.0
 * @date 2021-01-03
 * @copyright Copyright (c) 2021
 *
 * @brief An example file to demonstrate the use of the matrix class template and to test multithreading performance with various block sizes.
 */

#include <chrono>   // std::chrono::duration, std::chrono::duration_cast, std::chrono::steady_clock, std::chrono::time_point, std::milli
#include <cmath>    // std::log10
#include <cstdint>  // std::int_fast64_t, std::uint_fast64_t
#include <iostream> // std::cout
#include <iomanip>  // std::setw
#include <random>   // std::uniform_real_distribution
#include <thread>   // std::thread

#define ENABLE_MATRIX_MULTITHREADING
#include "matrix.hpp"
using matrices::matrix;

typedef std::int_fast64_t i64;
typedef std::uint_fast64_t ui64;

/**
 * @brief A class for measuring execution time.
 */
class timer
{
public:
    /**
     * @brief Start (or restart) measuring time.
     */
    void start()
    {
        start_time = std::chrono::steady_clock::now();
    }

    /**
     * @brief Stop measuring time and store the elapsed time since start().
     */
    void stop()
    {
        elapsed_time = std::chrono::steady_clock::now() - start_time;
    }

    /**
     * @brief Get the number of milliseconds that have elapsed between start() and stop().
     *
     * @return The number of milliseconds.
     */
    i64 ms() const
    {
        return (std::chrono::duration_cast<std::chrono::duration<i64, std::milli>>(elapsed_time)).count();
    }

private:
    /**
     * @brief The time point when measuring started.
     */
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

    /**
     * @brief The duration that has elapsed between start() and stop().
     */
    std::chrono::duration<double> elapsed_time = std::chrono::duration<double>::zero();
};

/**
 * @brief A function to test the execution time of selected matrix operations using different block sizes.
 *
 * @param rows The number of rows in the test matrices.
 * @param cols The number of columns in the test matrices.
 */
void measure_operations(const ui64 &rows, const ui64 &cols)
{
    // Initialize a matrix<double>::random_generator object to generates matrices with real (floating-point) numbers uniformly distributed between -1000 and 1000.
    matrix<double>::random_generator<std::uniform_real_distribution<double>> rnd(-1000, 1000);
    // Initialize a timer object to measure the execution time of various operations.
    timer tmr;
    // The number of available hardware threads. With a hyperthreaded CPU, this will be twice the number of CPU cores.
    const ui64 hardware_threads = std::thread::hardware_concurrency();
    // The total number of elements in the test matrices.
    const ui64 total_size = rows * cols;
    // The block sizes to try.
    const ui64 try_blocks[] = {total_size, total_size / hardware_threads * 4, total_size / hardware_threads * 2, total_size / hardware_threads, total_size / hardware_threads / 2, total_size / hardware_threads / 4};
    // The character width of the block numbers, for formatting purposes.
    const int width = (int)(std::log10(total_size)) + 1;

    // Generate four random test matrices. We first set the block size such that the number of blocks is equal to the number of threads, to ensure that this operation is optimized.
    matrix<double>::global_block_size = 0;
    matrix<double> R = rnd.generate_matrix(rows, cols);
    matrix<double> S = rnd.generate_matrix(rows, cols);
    matrix<double> T = rnd.generate_matrix(rows, cols);
    matrix<double> U = rnd.generate_matrix(rows, cols);

    std::cout << "\nAdding two " << rows << "x" << cols << " matrices (A = R + S):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        tmr.start();
        matrix<double> A = R + S;
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    std::cout << "\nAdding three " << rows << "x" << cols << " matrices (A = R + S + T):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        tmr.start();
        matrix<double> A = R + S + T;
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    std::cout << "\nAdding four " << rows << "x" << cols << " matrices (A = R + S + T + U):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        tmr.start();
        matrix<double> A = R + S + T + U;
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    std::cout << "\nAdding four " << rows << "x" << cols << " matrices with scalar coefficients\n(A = x * R + y * S + z * T + w * U):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        double x = rnd.generate_scalar();
        double y = rnd.generate_scalar();
        double z = rnd.generate_scalar();
        double w = rnd.generate_scalar();
        tmr.start();
        matrix<double> A = x * R + y * S + z * T + w * U;
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    std::cout << "\nGenerating random " << rows << "x" << cols << " matrix (rnd.randomize_matrix(A)):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        matrix<double> A(rows, cols);
        tmr.start();
        rnd.randomize_matrix(A);
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    std::cout << "\nTransposing one " << rows << "x" << cols << " matrix (A = R.transpose()):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n;
        tmr.start();
        matrix<double> A = R.transpose();
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }

    // Since matrix multiplication is O(n^3), we reduce the size of the test matrices here so that this operation completes within a reasonable time.
    constexpr ui64 mult_factor = 6;
    matrix<double>::global_block_size = 0;
    matrix<double> X = rnd.generate_matrix(rows / mult_factor, cols / mult_factor);
    matrix<double> Y = rnd.generate_matrix(cols / mult_factor, rows / mult_factor);
    std::cout << "\nMultiplying two " << rows / mult_factor << "x" << cols / mult_factor << " matrices (A = X * Y):\n";
    for (ui64 n : try_blocks)
    {
        matrix<double>::global_block_size = n / mult_factor / mult_factor;
        tmr.start();
        matrix<double> A = X * Y;
        tmr.stop();
        std::cout << "With block size of " << std::setw(width) << n / mult_factor / mult_factor << " (" << std::setw(2) << total_size / n << " blocks), execution took " << tmr.ms() << " ms.\n";
    }
}

int main()
{
    measure_operations(4800, 4800);
}
