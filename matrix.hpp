#pragma once

/**
 * @file matrix.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
 * @version 1.0
 * @date 2021-01-03
 * @copyright Copyright (c) 2021
 *
 * @brief A performance-oriented matrix class template. Please see the attached README.md file for more information and examples.
 */

#include <algorithm>        // std::max
#include <atomic>           // std::atomic
#include <cstdint>          // std::uint_fast32_t, std::uint_fast64_t
#include <functional>       // std::function
#include <initializer_list> // std::initializer_list
#include <iomanip>          // std::setw
#include <iostream>         // std::ostream, std::streamsize
#include <memory>           // std::unique_ptr
#include <mutex>            // std::mutex, std::scoped_lock
#include <queue>            // std::queue
#include <random>           // std::mt19937_64, std::random_device
#include <sstream>          // std::ostringstream
#include <thread>           // std::this_thread, std::thread
#include <type_traits>      // std::common_type_t, std::enable_if_t, std::is_base_of_v, std::remove_cv_t, std::remove_reference_t
#include <utility>          // std::move

namespace matrices
{
    typedef std::uint_fast32_t ui32;
    typedef std::uint_fast64_t ui64;

    // ==========
    // Exceptions
    // ==========

    /**
     * @brief Exception to be thrown when adding or subtracting two matrices if they do not have the same number of rows or columns.
     */
    class incompatible_sizes_add
    {
    };

    /**
     * @brief Exception to be thrown when multiplying two matrices if the number of columns in the first matrix is not the same as the number of rows in the second matrix.
     */
    class incompatible_sizes_multiply
    {
    };

    /**
     * @brief Exception to be thrown when accessing matrix elements using the at() function if the row or column index is out of range.
     */
    class index_out_of_range
    {
    };

    /**
     * @brief Exception to be thrown when using constructors that accept an std::initializer_list if the list's size is incompatible with the size of the matrix.
     */
    class initializer_wrong_size
    {
    };

    /**
     * @brief Exception to be thrown when taking the trace if the matrix is not square.
     */
    class not_square
    {
    };

    /**
     * @brief Exception to be thrown when using constructors if the desired number of rows or columns is zero.
     */
    class zero_size
    {
    };

    // ==========================================
    // Expression templates and related functions
    // ==========================================

    /**
     * @brief A class to derive both matrix and matrix_op from. Used to ensure that relevant templates only act on matrices and/or expression templates.
     */
    class matrix_base
    {
    };

    /**
     * @brief Check if a given type is either a matrix or an expression template. Will be used with std::enable_if_t in relevant templates.
     *
     * @tparam T The type to check.
     */
    template <typename T>
    constexpr bool is_matrix_or_op = std::is_base_of_v<matrix_base, std::remove_cv_t<std::remove_reference_t<T>>>;

    /**
     * @brief An expression template representing an element-by-element operation on matrices.
     *
     * @tparam F The type of the operation to perform. The operation should be a function which takes exactly one argument: the index of the element to perform the operation on.
     */
    template <typename F>
    class matrix_op : public matrix_base
    {
    public:
        matrix_op(const F &_func, const ui64 &_rows, const ui64 &_cols)
            : func(_func), rows(_rows), cols(_cols) {}

        ui64 get_rows() const
        {
            return rows;
        }

        ui64 get_cols() const
        {
            return cols;
        }

        auto operator[](const ui64 &i) const
        {
            return func(i);
        }

        auto operator()(const ui64 &row, const ui64 &col) const
        {
            return func((cols * row) + col);
        }

    private:
        const F func;
        const ui64 rows;
        const ui64 cols;
    };

    /**
     * @brief Add two matrices and/or expression templates. Can be multithreaded.
     *
     * @tparam A The type of the first object to add.
     * @tparam B The type of the second object to add.
     * @param a The first object to add.
     * @param b The second object to add.
     * @return An expression template representing the addition.
     * @throws incompatible_sizes_add if the objects do not have the same number of rows and columns.
     */
    template <typename A, typename B, typename = std::enable_if_t<is_matrix_or_op<A> and is_matrix_or_op<B>>>
    auto operator+(const A &a, const B &b)
    {
        if ((a.get_rows() != b.get_rows()) or (a.get_cols() != b.get_cols()))
            throw incompatible_sizes_add{};
        return matrix_op([&a, &b](const ui64 &i) { return a[i] + b[i]; }, a.get_rows(), a.get_cols());
    }

    /**
     * @brief Negate a matrix and/or expression template. Can be multithreaded.
     *
     * @tparam A The type of the object to negate.
     * @param a The object to negate.
     * @return An expression template representing the negation.
     */
    template <typename A, typename = std::enable_if_t<is_matrix_or_op<A>>>
    auto operator-(const A &a)
    {
        return matrix_op([&a](const ui64 &i) { return -a[i]; }, a.get_rows(), a.get_cols());
    }

    /**
     * @brief Subtract two matrices and/or expression templates. Can be multithreaded.
     *
     * @tparam A The type of the first object to subtract.
     * @tparam B The type of the second object to subtract.
     * @param a The first object to subtract.
     * @param b The second object to subtract.
     * @return An expression template representing the subtraction.
     * @throws incompatible_sizes_add if the objects do not have the same number of rows and columns.
     */
    template <typename A, typename B, typename = std::enable_if_t<is_matrix_or_op<A> and is_matrix_or_op<B>>>
    auto operator-(const A &a, const B &b)
    {
        if ((a.get_rows() != b.get_rows()) or (a.get_cols() != b.get_cols()))
            throw incompatible_sizes_add{};
        return matrix_op([&a, &b](const ui64 &i) { return a[i] - b[i]; }, a.get_rows(), a.get_cols());
    }

#ifdef ENABLE_MATRIX_MULTITHREADING
    ////////////////////////////////////////////////////////////////////////////////////////
    //                              Begin class thread_pool                               //
    //                                                                                    //

    /**
     * @brief A simple thread pool class. Maintains a queue of tasks, which are executed by threads in the pool as they become available.
     */
    class thread_pool
    {
    public:
        // ============================
        // Constructors and destructors
        // ============================

        /**
         * @brief Construct a new thread pool.
         *
         * @param _thread_count The number of threads to use. Default value is the total number of hardware threads available, as reported by the implementation. With a hyperthreaded CPU, this will be twice the number of CPU cores. If the argument is zero, 1 thread will be used.
         */
        thread_pool(const ui32 &_thread_count = std::thread::hardware_concurrency())
            : thread_count(std::max<ui32>(_thread_count, 1)), threads(new std::thread[std::max<ui32>(_thread_count, 1)])
        {
            for (ui32 i = 0; i < thread_count; i++)
            {
                threads[i] = std::thread(&thread_pool::worker, this);
            }
        }

        /**
         * @brief Destruct the thread pool. Waits for all submitted tasks to be completed, then destroys all threads.
         */
        ~thread_pool()
        {
            wait_for_tasks();
            running = false;
            for (ui32 i = 0; i < thread_count; i++)
            {
                threads[i].join();
            }
        }

        // =======================
        // Public member functions
        // =======================

        /**
         * @brief Get the number of threads in the pool.
         *
         * @return The number of threads.
         */
        ui32 get_thread_count() const
        {
            return thread_count;
        }

        /**
         * @brief Push a new task (a function to be executed) into the queue.
         *
         * @tparam F The type of the task.
         * @param task The task to push. Should be a function with no arguments or return value.
         */
        template <typename F>
        void push_task(const F &task)
        {
            tasks_running++;
            const std::scoped_lock lock(queue_mutex);
            tasks.push(std::move(std::function<void()>(task)));
        }

        /**
         * @brief Wait for all running tasks to be completed. Must be called before attempting to access any data that may have been modified by the tasks.
         */
        void wait_for_tasks()
        {
            while (tasks_running != 0)
            {
                std::this_thread::yield();
            }
        }

    private:
        // ========================
        // Private member functions
        // ========================

        /**
         * @brief Try to pop a new task out of the queue.
         *
         * @param task A reference to the task. Will be populated with a function if the queue is not empty.
         * @return true if a task was found, false if the queue is empty.
         */
        bool pop_task(std::function<void()> &task)
        {
            const std::scoped_lock lock(queue_mutex);
            if (tasks.empty())
                return false;
            task = std::move(tasks.front());
            tasks.pop();
            return true;
        }

        /**
         * @brief A worker function to be assigned to each thread in the pool. Loops forever and executes any task assigned to it, until the atomic variable running is set to false by the destructor.
         */
        void worker()
        {
            while (running)
            {
                std::function<void()> task;
                if (pop_task(task))
                {
                    task();
                    tasks_running--;
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        }

        // ============
        // Private data
        // ============

        /**
         * @brief An atomic variable indicating to the threads to keep running.
         */
        std::atomic<bool> running = true;

        /**
         * @brief An atomic variable to keep track of how many tasks are currently running.
         */
        std::atomic<ui32> tasks_running = 0;

        /**
         * @brief A mutex to synchronize access to the task queue by different threads.
         */
        mutable std::mutex queue_mutex;

        /**
         * @brief A queue of tasks to be executed by the threads.
         */
        std::queue<std::function<void()>> tasks;

        /**
         * @brief The number of threads in the pool.
         */
        const ui32 thread_count;

        /**
         * @brief A smart pointer to manage the memory allocated for the threads.
         */
        std::unique_ptr<std::thread[]> threads;
    };

    //                                                                                    //
    //                               End class thread_pool                                //
    ////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////
    //                              Begin class parallelizer                              //
    //                                                                                    //

    /**
     * @brief A class to automatically divide a large number of element-wise or row-wise operations on a matrix into multiple tasks to be executed in parallel.
     */
    class parallelizer
    {
    public:
        // ============================
        // Constructors and destructors
        // ============================

        /**
         * @brief Construct a new parallelizer.
         *
         * @param _block_size  The size of the blocks to process in parallel. If set to 0, the block size will be the number of elements in the matrix divided by the number of threads in the thread pool.
         * @param _rows The number of rows in the matrix.
         * @param _cols The number of columns in the matrix.
         */
        parallelizer(const ui64 &_block_size, const ui64 &_rows, const ui64 &_cols)
            : block_size(_block_size), rows(_rows), cols(_cols)
        {
            divide_blocks();
            divide_rows();
        }

        /**
         * @brief Destruct the parallelizer. Waits for all submitted tasks to be completed.
         */
        ~parallelizer()
        {
            wait_for_my_tasks();
        }

        // =======================
        // Public member functions
        // =======================

        /**
         * @brief Wait for all tasks submitted to the pool by this parallelizer to be completed.
         */
        void wait_for_my_tasks()
        {
            while (blocks_running != 0)
            {
                std::this_thread::yield();
            }
        }

        /**
         * @brief Change the block size for parallelization, and redistribute the blocks.
         *
         * @param _block_size The desired block size. If set to 0, the block size will be the number of elements in the matrix divided by the number of threads in the thread pool.
         */
        void set_block_size(const ui64 &_block_size)
        {
            block_size = _block_size;
            divide_blocks();
            divide_rows();
        }

        /**
         * @brief Change the number of rows and columns for parallelization, and redistribute the blocks.
         *
         * @param _rows The number of rows in the matrix.
         * @param _cols The number of columns in the matrix.
         */
        void set_rows_cols(const ui64 &_rows, const ui64 &_cols)
        {
            rows = _rows;
            cols = _cols;
            divide_blocks();
            divide_rows();
        }

        /**
         * @brief Perform an arbitrary task element-by-element in parallel blocks, looping over the elements in each block and executing the task for each element individually.
         * @details The task should be a function which takes one parameter: the index of the element to operate on. This type of parallelization is used in basic operations, such as addition, which are performed element-wise without accessing any external resources.
         *
         * @tparam F The type of the task.
         * @param task The task to perform.
         */
        template <typename F>
        void parallelize_index(const F &task)
        {
            for (ui64 i = 0; i < num_linear_blocks; i++)
            {
                blocks_running++;
                pool.push_task([&start = linear_ranges[i].start, &end = linear_ranges[i].end, &task, this] {
                    for (ui64 j = start; j <= end; j++)
                        task(j);
                    blocks_running--;
                });
            }
            wait_for_my_tasks();
        }

        /**
         * @brief Perform an arbitrary task element-by-element in parallel blocks, passing the start and end indices (inclusive) of each block to the task and leaving it up to the task itself to decide how to act on individual elements within the blocks.
         * @details The task should be a function which takes two parameters: the start and end indices (inclusive) of the elements to operate on, as determined by the parallelizer. This type of parallelization is used when we want to allocate a separate external resource for each block so that they don't all try to access the same resource at once, such as in the random_generator class.
         *
         * @tparam F The type of the task.
         * @param task The task to perform.
         */
        template <typename F>
        void parallelize_start_end(const F &task)
        {
            for (ui64 i = 0; i < num_linear_blocks; i++)
            {
                blocks_running++;
                pool.push_task([&start = linear_ranges[i].start, &end = linear_ranges[i].end, &task, this] {
                    task(start, end);
                    blocks_running--;
                });
            }
            wait_for_my_tasks();
        }

        /**
         * @brief Perform an arbitrary task row-by-row in blocks, passing the start and end rows (inclusive) of each block to the task and leaving it up to the task itself to decide how to act on individual elements within the blocks.
         * @details The task should be a function which takes two parameters: the start and end rows (inclusive) to operate on, as determined by the parallelizer. This type of parallelization is used in operations that are performed row-by-row instead of element-by-element, such as transposition and multiplication.
         *
         * @tparam F The type of the task.
         * @param task The task to perform.
         */
        template <typename F>
        void parallelize_by_row(const F &task)
        {
            for (ui64 i = 0; i < num_row_blocks; i++)
            {
                blocks_running++;
                pool.push_task([&start = row_ranges[i].start, &end = row_ranges[i].end, &task, this] {
                    task(start, end);
                    blocks_running--;
                });
            }
            wait_for_my_tasks();
        }

    private:
        // ========================
        // Private member functions
        // ========================

        /**
         * @brief Divide the elements of the matrix into blocks. The array linear_ranges will store the start and end indices (inclusive) of each block.
         * @details For example, a 100x100 matrix with a block size of 2000 will be divided into 5 blocks with 2000 elements each: 0-1999, 2000-3999, 4000-5999, 6000-7999, 8000-9999. The size of the last block will generally be smaller than the desired block size, unless the number of elements in the matrix divides the block size exactly.
         */
        void divide_blocks()
        {
            if (rows == 0 or cols == 0)
            {
                num_linear_blocks = 0;
                linear_ranges.reset(nullptr);
            }
            else
            {
                ui64 use_block_size = block_size ? block_size : std::max<ui64>(rows * cols / pool.get_thread_count(), 1);
                num_linear_blocks = rows * cols / use_block_size;
                if ((rows * cols) - (num_linear_blocks * use_block_size) != 0)
                    num_linear_blocks++;
                if (num_linear_blocks == 0)
                    num_linear_blocks = 1;
                linear_ranges.reset(new block_ranges[num_linear_blocks]);
                for (ui64 i = 0; i < num_linear_blocks; i++)
                {
                    linear_ranges[i].start = i * use_block_size;
                    if (i == num_linear_blocks - 1)
                        linear_ranges[i].end = (rows * cols) - 1;
                    else
                        linear_ranges[i].end = ((i + 1) * use_block_size) - 1;
                }
            }
        }

        /**
         * @brief Divide the rows of the matrix into blocks based on the desired block size and the length of each row (i.e. the number of columns). The array row_ranges will store the start and end rows (inclusive) of each block.
         * @details For example, a 100x100 matrix with a block size of 2000 will be divided into 5 blocks with 20 rows each: 0-19, 20-39, 40-59, 60-79, 80-99. Note that each row must be fully contained in a single block, so if the row length is larger than the desired block size, the block size will be enlarged to accommodate the full rows.
         */
        void divide_rows()
        {
            if (rows == 0 or cols == 0)
            {
                num_row_blocks = 0;
                row_ranges.reset(nullptr);
            }
            else
            {
                ui64 use_block_size = block_size ? block_size : std::max<ui64>(rows * cols / pool.get_thread_count(), 1);
                ui64 rows_per_block = use_block_size / cols;
                if (rows_per_block == 0)
                {
                    rows_per_block = 1;
                    num_row_blocks = rows;
                }
                else
                    num_row_blocks = rows / rows_per_block;
                if (num_row_blocks == 0)
                    num_row_blocks = 1;
                row_ranges.reset(new block_ranges[num_row_blocks]);
                for (ui64 i = 0; i < num_row_blocks; i++)
                {
                    row_ranges[i].start = i * rows_per_block;
                    if (i == num_row_blocks - 1)
                        row_ranges[i].end = rows - 1;
                    else
                        row_ranges[i].end = ((i + 1) * rows_per_block) - 1;
                }
            }
        }

        // ========================
        // Private member variables
        // ========================

        /**
         * @brief A structure to store the start and end positions of each block.
         */
        struct block_ranges
        {
            ui64 start;
            ui64 end;
        };

        /**
         * @brief An atomic variable to keep track of how many block tasks submitted by this parallelizer are currently running.
         */
        std::atomic<ui64> blocks_running = 0;

        /**
         * @brief The size of blocks to process in parallel. If set to 0, the block size will be the number of elements in the matrix divided by the number of threads in the thread pool.
         */
        ui64 block_size;

        /**
         * @brief The number of rows in the matrix.
         */
        ui64 rows;

        /**
         * @brief The number of columns in the matrix.
         */
        ui64 cols;

        /**
         * @brief The number of blocks that linear operations have been divided into.
         */
        ui64 num_linear_blocks;

        /**
         * @brief The number of blocks that row operations have been divided into.
         */
        ui64 num_row_blocks;

        /**
         * @brief A smart pointer to manage the memory allocated for the linear block ranges.
         */
        std::unique_ptr<block_ranges[]> linear_ranges;

        /**
         * @brief A smart pointer to manage the memory allocated for the row block ranges.
         */
        std::unique_ptr<block_ranges[]> row_ranges;

        // ===============================
        // Static private member variables
        // ===============================

        /**
         * @brief A thread pool object to use for executing tasks. The same pool will be used by all parallelizers.
         */
        inline static thread_pool pool;
    };
#endif

    //                                                                                    //
    //                               End class parallelizer                               //
    ////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////
    //                                 Begin class matrix                                 //
    //                                                                                    //

    /**
     * @brief A performance-oriented matrix class template.
     *
     * @tparam T The type to use for the matrix elements. Can be any type that has addition, subtraction, negation, and multiplication defined.
     */
    template <typename T>
    class matrix : public matrix_base
    {
    public:
        // ============
        // Constructors
        // ============

        /**
         * @brief Construct a degenerate matrix with zero rows and columns. Should not be used in practice; only exists so that the class has a default constructor, e.g. for the purpose of creating an array of matrices and constructing each matrix later.
         */
        matrix()
            : rows(0), cols(0), smart_elements(new T[0])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, 0, 0)
#endif
        {
            elements = smart_elements.get();
        }

        /**
         * @brief Construct an uninitialized matrix. Don't forget to initialize the elements before accessing them!
         *
         * @param _rows The number of rows.
         * @param _cols The number of columns.
         * @throws zero_size if the number of rows or columns is zero.
         */
        matrix(const ui64 &_rows, const ui64 &_cols)
            : rows(_rows), cols(_cols), smart_elements(new T[_rows * _cols])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, _rows, _cols)
#endif
        {
            if (rows == 0 or cols == 0)
                throw zero_size{};
            elements = smart_elements.get();
        }

        /**
         * @brief Construct a matrix with all of its elements initialized to the same scalar. Can be multithreaded.
         *
         * @param _rows The number of rows.
         * @param _cols The number of columns.
         * @param scalar The scalar to initialize all of the elements to.
         * @throws zero_size if the number of rows or columns is zero.
         */
        matrix(const ui64 &_rows, const ui64 &_cols, const T &scalar)
            : rows(_rows), cols(_cols), smart_elements(new T[_rows * _cols])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, _rows, _cols)
#endif
        {
            if (rows == 0 or cols == 0)
                throw zero_size{};
            elements = smart_elements.get();
            fill(scalar);
        }

        /**
         * @brief Construct a diagonal matrix and initialize its diagonal using an array. Can be multithreaded.
         *
         * @param dim The number of rows and columns.
         * @param diagonal An array containing the elements on the diagonal.
         * @throws zero_size if the number of rows or columns is zero.
         */
        matrix(const ui64 &dim, const T *diagonal)
            : rows(dim), cols(dim), smart_elements(new T[dim * dim])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, dim, dim)
#endif
        {
            if (dim == 0)
                throw zero_size{};
            elements = smart_elements.get();
            fill(0);
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.wait_for_my_tasks();
#endif
            for (ui64 i = 0; i < rows; i++)
                operator()(i, i) = diagonal[i];
        }

        /**
         * @brief Construct a diagonal matrix and initialize its diagonal using an std::initializer_list. Can be multithreaded.
         *
         * @param dim The number of rows and columns.
         * @param diagonal An std::initializer_list containing the elements on the diagonal.
         * @throws initializer_wrong_size if the size of the std::initializer_list does not equal the number of rows and columns.
         * @throws zero_size if the number of rows and columns is zero.
         */
        matrix(const ui64 &dim, const std::initializer_list<T> &diagonal)
            : rows(dim), cols(dim), smart_elements(new T[dim * dim])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, dim, dim)
#endif
        {
            if (dim == 0)
                throw zero_size{};
            if (diagonal.size() != dim)
                throw initializer_wrong_size{};
            elements = smart_elements.get();
            fill(0);
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.wait_for_my_tasks();
#endif
            for (ui64 i = 0; i < rows; i++)
                operator()(i, i) = diagonal.begin()[i];
        }

        /**
         * @brief Construct a matrix and initialize it using an std::initializer_list. Can be multithreaded.
         * @details The elements should be given in flattened 1-dimensional form, in row-major order. This means that the matrix element at row i and column j is given by element number (cols * i) + j of the std::initializer_list. For example, for a 2x2 matrix A, the std::initializer_list will be {A(0, 0), A(0, 1), A(1, 0), A(1, 1)}.
         *
         * @param _rows The number of rows.
         * @param _cols The number of columns.
         * @param _elements An std::initializer_list containing the elements in flattened 1-dimensional form, in row-major order.
         * @throws initializer_wrong_size if the size of the std::initializer_list does not equal the total number of matrix elements.
         * @throws zero_size if the number of rows or columns is zero.
         */
        matrix(const ui64 &_rows, const ui64 &_cols, const std::initializer_list<T> &_elements)
            : rows(_rows), cols(_cols), smart_elements(new T[_rows * _cols])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, _rows, _cols)
#endif
        {
            if (rows == 0 or cols == 0)
                throw zero_size{};
            if (_elements.size() != rows * cols)
                throw initializer_wrong_size{};
            elements = smart_elements.get();
            copy(_elements.begin(), elements);
        }

        /**
         * @brief Construct a new matrix by copying the elements of an existing matrix. Can be multithreaded.
         *
         * @param m The matrix to be copied.
         */
        matrix(const matrix<T> &m)
            : rows(m.rows), cols(m.cols), smart_elements(new T[m.rows * m.cols])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, m.rows, m.cols)
#endif
        {
            elements = smart_elements.get();
            copy(m.elements, elements);
        }

        /**
         * @brief Construct a new matrix by moving the elements of an existing matrix. The source matrix will become degenerate, with zero rows and columns and no elements.
         *
         * @param m The matrix to be moved.
         */
        matrix(matrix<T> &&m)
            : rows(m.rows), cols(m.cols), smart_elements(std::move(m.smart_elements))
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, m.rows, m.cols)
#endif
        {
            elements = smart_elements.get();
            m.rows = 0;
            m.cols = 0;
            m.elements = nullptr;
        }

        /**
         * @brief Construct a matrix based on the result of evaluating an expression template. Can be multithreaded.
         *
         * @tparam E The type of the expression template to evaluate.
         * @param e The expression template to evaluate.
         */
        template <typename E, typename = std::enable_if_t<is_matrix_or_op<E>>>
        matrix(const E &e)
            : rows(e.get_rows()), cols(e.get_cols()), smart_elements(new T[e.get_rows() * e.get_cols()])
#ifdef ENABLE_MATRIX_MULTITHREADING
              ,
              prl(global_block_size, e.get_rows(), e.get_cols())
#endif
        {
            elements = smart_elements.get();
            operator=(e);
        }

        // =======================
        // Public member functions
        // =======================

        /**
         * @brief Set all the elements of the matrix to the same scalar. Can be multithreaded.
         *
         * @param scalar The scalar.
         */
        void fill(const T &scalar)
        {
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.parallelize_index([this, &scalar](const ui64 &i) {
#else
            for (ui64 i = 0; i < rows * cols; i++)
#endif
                elements[i] = scalar;
#ifdef ENABLE_MATRIX_MULTITHREADING
            });
#endif
        }

        /**
         * @brief Copy an array and use its elements as the elements of this matrix. Can be multithreaded.
         * @details The elements should be given in flattened 1-dimensional form, in row-major order. This means that the matrix element at row i and column j is given by element number (cols * i) + j of the array. For example, for a 2x2 matrix A, the array will be {A(0, 0), A(0, 1), A(1, 0), A(1, 1)}.
         *
         * @param source The array to copy.
         */
        void from_array(const T *source)
        {
            copy(source, elements);
        }

        /**
         * @brief Get the number of columns in the matrix.
         *
         * @return The number of columns.
         */
        ui64 get_cols() const
        {
            return cols;
        }

        /**
         * @brief Get the number of rows in the matrix.
         *
         * @return The number of rows.
         */
        ui64 get_rows() const
        {
            return rows;
        }

        /**
         * @brief Take the trace of the matrix, i.e. the sum of elements on the diagonal.
         *
         * @param init The initial value of the sum. Default is 0. Can be changed if the elements are of a user-defined type that has an additive identity other than the number 0.
         * @return The trace.
         * @throws not_square if the matrix is not a square matrix.
         */
        T trace(const T &init = 0) const
        {
            if (rows != cols)
                throw not_square{};
            T tr = init;
            for (ui64 i = 0; i < rows; i++)
                tr += operator()(i, i);
            return tr;
        }

        /**
         * @brief Transpose a matrix. Can be multithreaded.
         *
         * @return The transposed matrix.
         */
        matrix<T> transpose() const
        {
            matrix<T> out(cols, rows);
#ifdef ENABLE_MATRIX_MULTITHREADING
            out.prl.parallelize_by_row([this, &out](const ui64 &start, const ui64 &end) {
                for (ui64 i = start; i <= end; i++)
#else
            for (ui64 i = 0; i < out.rows; i++)
#endif
                    for (ui64 j = 0; j < out.cols; j++)
                        out(i, j) = operator()(j, i);
#ifdef ENABLE_MATRIX_MULTITHREADING
            });
#endif
            return out;
        }

        // ===============================
        // Overloaded assignment operators
        // ===============================

        /**
         * @brief Copy the elements of another matrix to this matrix. Can be multithreaded.
         *
         * @param m The matrix to be copied.
         * @return A reference to this matrix.
         */
        matrix<T> &operator=(const matrix<T> &m)
        {
            rows = m.rows;
            cols = m.cols;
            smart_elements.reset(new T[rows * cols]);
            elements = smart_elements.get();
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.set_rows_cols(rows, cols);
#endif
            copy(m.elements, elements);
            return *this;
        }

        /**
         * @brief Move the elements of another matrix to this matrix. The source matrix will become degenerate, with zero rows and columns and no elements.
         *
         * @param m The matrix to be moved.
         * @return A reference to this matrix.
         */
        matrix<T> &operator=(matrix<T> &&m)
        {
            rows = m.rows;
            cols = m.cols;
            smart_elements = std::move(m.smart_elements);
            elements = smart_elements.get();
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.set_rows_cols(rows, cols);
            m.prl.set_rows_cols(0, 0);
#endif
            m.rows = 0;
            m.cols = 0;
            m.elements = nullptr;
            return *this;
        }

        /**
         * @brief Evaluate an expression template and store the result in this matrix. A large number of expression templates can be evaluated with just one loop, potentially providing a significant performance boost. Can be multithreaded.
         *
         * @tparam E The type of the expression template to evaluate.
         * @param e The expression template to evaluate.
         * @return A reference to this matrix.
         */
        template <typename E, typename = std::enable_if_t<is_matrix_or_op<E>>>
        matrix<T> &operator=(const E &e)
        {
            if (rows != e.get_rows() or cols != e.get_cols())
            {
                rows = e.get_rows();
                cols = e.get_cols();
                smart_elements.reset(new T[rows * cols]);
                elements = smart_elements.get();
            }
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.parallelize_index([this, &e](const ui64 &i) {
#else
            for (ui64 i = 0; i < rows * cols; i++)
#endif
                elements[i] = e[i];
#ifdef ENABLE_MATRIX_MULTITHREADING
            });
#endif
            return *this;
        }

        /**
         * @brief Evaluate an expression template and add the result to this matrix. Can be multithreaded.
         *
         * @tparam E The type of the expression template to evaluate.
         * @param e The expression template to evaluate.
         * @return A reference to this matrix.
         * @throws incompatible_sizes_add if the matrices do not have the same number of rows and columns.
         */
        template <typename E, typename = std::enable_if_t<is_matrix_or_op<E>>>
        matrix<T> &operator+=(const E &e)
        {
            *this = *this + e;
            return *this;
        }

        /**
         * @brief Evaluate an expression template and subtract the result from this matrix. Can be multithreaded.
         *
         * @tparam E The type of the expression template to evaluate.
         * @param e The expression template to evaluate.
         * @return A reference to this matrix.
         * @throws incompatible_sizes_add if the matrices do not have the same number of rows and columns.
         */
        template <typename E, typename = std::enable_if_t<is_matrix_or_op<E>>>
        matrix<T> &operator-=(const E &e)
        {
            *this = *this - e;
            return *this;
        }

        /**
         * @brief Multiply this matrix with a scalar on the right and assign the result to this matrix. Can be multithreaded.
         * @details Note that there is no version of the assignment multiplication operator for multiplying by a scalar from the left. If multiplication is non-commutative and multiplication by a scalar from the left is desired, use the regular multiplication operator instead.
         *
         * @param scalar The scalar.
         * @return A reference to this matrix.
         */
        matrix<T> operator*=(const T &scalar)
        {
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.parallelize_index([this, &scalar](const ui64 &i) {
#else
            for (ui64 i = 0; i < rows * cols; i++)
#endif
                elements[i] = elements[i] * scalar;
#ifdef ENABLE_MATRIX_MULTITHREADING
            });
#endif
            return *this;
        }

        /**
         * @brief Evaluate an expression template and multiply this matrix by the result. Can be multithreaded.
         *
         * @tparam E The type of the expression template to evaluate.
         * @param e The expression template to evaluate.
         * @return A reference to this matrix.
         * @throws incompatible_sizes_multiply if the number of columns in the first matrix is not the same as the number of rows in the second matrix.
         */
        template <typename E, typename = std::enable_if_t<is_matrix_or_op<E>>>
        matrix<T> &operator*=(const E &e)
        {
            *this = *this * e;
            return *this;
        }

        // =====================================================
        // Functions and overloaded operators for element access
        // =====================================================

        /**
         * @brief Read or modify a matrix element, without range checking.
         *
         * @param row The row index (starting from zero).
         * @param col The column index (starting from zero).
         * @return A reference to the element.
         */
        T &operator()(const ui64 &row, const ui64 &col)
        {
            return elements[(cols * row) + col];
        }

        /**
         * @brief Read a matrix element, without range checking.
         *
         * @param row The row index (starting from zero).
         * @param col The column index (starting from zero).
         * @return The value of the element.
         */
        T operator()(const ui64 &row, const ui64 &col) const
        {
            return elements[(cols * row) + col];
        }

        /**
         * @brief Read or modify an element of the underlying 1-dimensional array, without range checking.
         *
         * @param i The element index (starting from zero).
         * @return A reference to the element.
         */
        T &operator[](const ui64 &i)
        {
            return elements[i];
        }

        /**
         * @brief Read an element of the underlying 1-dimensional array, without range checking.
         *
         * @param i The element index (starting from zero).
         * @return The value of the element.
         */
        T operator[](const ui64 &i) const
        {
            return elements[i];
        }

        /**
         * @brief Read or modify a matrix element, with range checking.
         *
         * @param row The row index (starting from zero).
         * @param col The column index (starting from zero).
         * @return A reference to the element.
         * @throws index_out_of_range if the requested row or column index is out of range.
         */
        T &at(const ui64 &row, const ui64 &col)
        {
            if (row >= rows or col >= cols)
                throw index_out_of_range{};
            return elements[(cols * row) + col];
        }

        /**
         * @brief Read a matrix element, with range checking.
         *
         * @param row The row index (starting from zero).
         * @param col The column index (starting from zero).
         * @return The value of the element.
         * @throws index_out_of_range if the requested row or column index is out of range.
         */
        T at(const ui64 &row, const ui64 &col) const
        {
            if (row >= rows or col >= cols)
                throw index_out_of_range{};
            return elements[(cols * row) + col];
        }

        // ================
        // Friend functions
        // ================

        /**
         * @brief Multiply a scalar on the left with a matrix or expression template on the right. Can be multithreaded.
         *
         * @tparam M The type of the matrix or expression template.
         * @param scalar The scalar.
         * @param m The matrix or expression template.
         * @return An expression template representing the multiplication.
         */
        template <typename M, typename = std::enable_if_t<is_matrix_or_op<M>>>
        friend auto operator*(const T &scalar, const M &m)
        {
            return matrix_op([&scalar, &m](const ui64 &i) { return scalar * m[i]; }, m.get_rows(), m.get_cols());
        }

        /**
         * @brief Multiply a matrix or expression template on the left with a scalar on the right. Can be multithreaded.
         *
         * @tparam M The type of the matrix or expression template.
         * @param m The matrix or expression template.
         * @param scalar The scalar.
         * @return An expression template representing the multiplication.
         */
        template <typename M, typename = std::enable_if_t<is_matrix_or_op<M>>>
        friend auto operator*(const M &m, const T &scalar)
        {
            return matrix_op([&m, &scalar](const ui64 &i) { return m[i] * scalar; }, m.get_rows(), m.get_cols());
        }

        // ==============================
        // Static public member variables
        // ==============================

#ifdef ENABLE_MATRIX_MULTITHREADING
        /**
         * @brief The global block size that will be used for parallelization of newly created matrices. Changing it will not affect the block sizes of any existing matrices. Default value is 0, which means that the block size will be the number of elements in the matrix divided by the number of threads in the thread pool.
         * @details Too small blocks would mean too many tasks will be submitted to the thread pool, leading to a slowdown due to the overhead in submitting each task. Too large blocks would mean no parallelization will take place.
         */
        inline static ui64 global_block_size = 0;
#endif

        // =====================
        // Public nested classes
        // =====================

        //////////////////////////////////////////////////////////////////
        //              Begin nested class random_generator             //
        //                                                              //

        /**
         * @brief A class template for generating random matrices.
         *
         * @tparam D The distribution to use, e.g. std::uniform_real_distribution<double>.
         */
        template <typename D>
        class random_generator
        {
        public:
            // ============
            // Constructors
            // ============

            /**
             * @brief Construct a new random matrix generator.
             *
             * @tparam P The types of the parameters to pass to the constructor of the distribution.
             * @param params The parameters to pass to the constructor of the distribution. The number of parameters and their types depends on the particular distribution being used.
             */
            template <typename... P>
            random_generator(P... params) : dist(params...) {}

            // =======================
            // Public member functions
            // =======================

            /**
             * @brief Generate a random scalar.
             *
             * @return The scalar.
             */
            T generate_scalar()
            {
                static std::mt19937_64 mt(rd());
                return dist(mt);
            }

            /**
             * @brief Generate a random matrix with the given number of rows and columns. Can be multithreaded.
             *
             * @param rows The desired number of rows in the matrix.
             * @param cols The desired number of columns in the matrix.
             * @return The random matrix.
             */
            matrix<T> generate_matrix(const ui64 &rows, const ui64 &cols)
            {
                matrix<T> m(rows, cols);
                randomize_matrix(m);
                return m;
            }

            /**
             * @brief Randomize a given matrix. Can be multithreaded.
             * @details Note that here we are using parallelize_start_end() and allocating a different std::mt19937_64 random number generator to each block, as sharing the same generator between the blocks would result in a very significant performance penalty.
             *
             * @param m The matrix to randomize.
             */
            void randomize_matrix(matrix<T> &m)
            {
#ifdef ENABLE_MATRIX_MULTITHREADING
                m.prl.parallelize_start_end([this, &m](const ui64 &start, const ui64 &end) {
                    std::mt19937_64 mt(generate_seed());
                    for (ui64 i = start; i <= end; i++)
#else
                static std::mt19937_64 mt(rd());
                for (ui64 i = 0; i < m.rows * m.cols; i++)
#endif
                        m.elements[i] = dist(mt);
#ifdef ENABLE_MATRIX_MULTITHREADING
                });
#endif
            }

        private:
            // ========================
            // Private member functions
            // ========================

#ifdef ENABLE_MATRIX_MULTITHREADING
            /**
             * @brief Generate a seed. The std::mt19937_64 in each thread will be seeded using generate_seed in order to avoid depleting the entropy of the random_device.
             *
             * @return A random unsigned 64-bit integer.
             */
            ui64 generate_seed()
            {
                static std::mt19937_64 mt(rd());
                return mt();
            }
#endif

            // ========================
            // Private member variables
            // ========================

            /**
             * @brief The distribution to use for generating random numbers.
             */
            D dist;

            /**
             * @brief The random device (hopefully a true random number generator) to be used for seeding the pseudo-random number generators.
             */
            std::random_device rd;
        };

        //                                                              //
        //               End nested class random_generator              //
        //////////////////////////////////////////////////////////////////

    private:
        // ========================
        // Private member functions
        // ========================

        /**
         * @brief Copy elements from one array to another. Can be multithreaded.
         *
         * @param in A pointer to the source array.
         * @param out A pointer to the target array. If both pointers are equal, no copying is performed.
         */
        void copy(const T *in, T *out)
        {
            if (in == out)
                return;
#ifdef ENABLE_MATRIX_MULTITHREADING
            prl.parallelize_index([in, out](const ui64 &i) {
#else
            for (ui64 i = 0; i < rows * cols; i++)
#endif
                out[i] = in[i];
#ifdef ENABLE_MATRIX_MULTITHREADING
            });
#endif
        }

        // ========================
        // Private member variables
        // ========================

        /**
         * @brief The number of rows.
         */
        ui64 rows = 0;

        /**
         * @brief The number of columns.
         */
        ui64 cols = 0;

        /**
         * @brief A pointer to an array used to store the elements of the matrix in flattened 1-dimensional form, in row-major order.
         */
        T *elements = nullptr;

        /**
         * @brief A smart pointer to manage the memory allocated for the matrix elements.
         */
        std::unique_ptr<T[]> smart_elements;

#ifdef ENABLE_MATRIX_MULTITHREADING
        /**
         * @brief The parallelizer to use for parallelizing operations on this matrix object.
         */
        parallelizer prl;
#endif

        // Define multiply_with_init as a friend function so it can use the parallelizer.
        template <typename A, typename B, typename S>
        friend auto multiply_with_init(const matrix<A> &a, const matrix<B> &b, const S &init);
    };

    //                                                                                    //
    //                                  End class matrix                                  //
    ////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Multiply two matrices, with a specific initial value for each element of the result. Should be used if the elements are of a user-defined type that has an additive identity other than the number 0. Can be multithreaded.
     *
     * @tparam A The type of the first matrix to multiply.
     * @tparam B The type of the second matrix to multiply.
     * @tparam S The type of the initial value.
     * @param a The first matrix to multiply.
     * @param b The second matrix to multiply.
     * @param init The initial value for each element of the product (should be the additive identity).
     * @return The product of the matrices.
     * @throws incompatible_sizes_multiply if the number of columns in the first matrix is not the same as the number of rows in the second matrix.
     */
    template <typename A, typename B, typename S>
    auto multiply_with_init(const matrix<A> &a, const matrix<B> &b, const S &init)
    {
        if (a.get_cols() != b.get_rows())
            throw incompatible_sizes_multiply{};
        matrix<std::common_type_t<A, B, S>> out(a.get_rows(), b.get_cols());
#ifdef ENABLE_MATRIX_MULTITHREADING
        out.prl.parallelize_by_row([&a, &b, &out, &init](const ui64 &start, const ui64 &end) {
            for (ui64 i = start; i <= end; i++)
#else
        for (ui64 i = 0; i < out.get_rows(); i++)
#endif
                for (ui64 j = 0; j < out.get_cols(); j++)
                {
                    out(i, j) = init;
                    for (ui64 k = 0; k < a.get_cols(); k++)
                        out(i, j) += a(i, k) * b(k, j);
                }
#ifdef ENABLE_MATRIX_MULTITHREADING
        });
#endif
        return out;
    }

    /**
     * @brief Multiply two matrices. Can be multithreaded.
     *
     * @tparam A The type of the first matrix to multiply.
     * @tparam B The type of the second matrix to multiply.
     * @param a The first matrix to multiply.
     * @param b The second matrix to multiply.
     * @return The product of the matrices.
     * @throws incompatible_sizes_multiply if the number of columns in the first matrix is not the same as the number of rows in the second matrix.
     */
    template <typename A, typename B>
    auto operator*(const matrix<A> &a, const matrix<B> &b)
    {
        return multiply_with_init(a, b, 0);
    }

    /**
     * @brief Print out a formatted matrix or expression template to a stream. The output will be formatted to visually look like a matrix. The character width of each element will be determined automatically (in a type-agnostic way) to ensure that the matrix is aligned correctly.
     *
     * @param out The output stream.
     * @param m The matrix to be printed.
     * @return A reference to the output stream.
     */
    template <typename M, typename = std::enable_if_t<is_matrix_or_op<M>>>
    std::ostream &operator<<(std::ostream &out, const M &m)
    {
        if (m.get_rows() == 0 and m.get_cols() == 0)
            out << "()\n";
        else
        {
            std::ostringstream ss;
            ui64 max_width = 0;
            for (ui64 i = 0; i < m.get_rows(); i++)
                for (ui64 j = 0; j < m.get_cols(); j++)
                {
                    ss << m(i, j);
                    max_width = std::max(max_width, ss.str().size());
                    ss.str("");
                }
            out << '\n';
            for (ui64 i = 0; i < m.get_rows(); i++)
            {
                out << "( ";
                for (ui64 j = 0; j < m.get_cols(); j++)
                    out << std::setw((int)max_width) << m(i, j) << ' ';
                out << ")\n";
            }
        }
        return out;
    }

} // namespace matrices
