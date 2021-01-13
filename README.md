<a id="markdown-a-fast-lightweight-easy-to-use-multithreaded-c17-matrix-class-template" name="a-fast-lightweight-easy-to-use-multithreaded-c17-matrix-class-template"></a>
# A fast, lightweight, easy-to-use, multithreaded C++17 matrix class template

<!-- TOC -->

- [A fast, lightweight, easy-to-use, multithreaded C++17 matrix class template](#a-fast-lightweight-easy-to-use-multithreaded-c17-matrix-class-template)
    - [Introduction](#introduction)
    - [Features](#features)
    - [Usage](#usage)
        - [Including the library](#including-the-library)
        - [Template parameters](#template-parameters)
        - [Constructors](#constructors)
        - [Overloaded operators and member functions](#overloaded-operators-and-member-functions)
        - [Printing matrices to streams](#printing-matrices-to-streams)
        - [Vectors](#vectors)
        - [Generating random matrices](#generating-random-matrices)
        - [Types with an additive identity different from zero](#types-with-an-additive-identity-different-from-zero)
        - [Degenerate matrices](#degenerate-matrices)
    - [Expression templates](#expression-templates)
        - [How expression templates work](#how-expression-templates-work)
        - [Forcing evaluation of expression templates](#forcing-evaluation-of-expression-templates)
    - [Multithreading](#multithreading)
        - [Enabling multithreading](#enabling-multithreading)
        - [How multithreading works, and when to use it](#how-multithreading-works-and-when-to-use-it)
        - [Setting a custom block size](#setting-a-custom-block-size)
        - [Using single-threaded and multithreaded matrices simultaneously](#using-single-threaded-and-multithreaded-matrices-simultaneously)
    - [Compiling](#compiling)
    - [Performance tests](#performance-tests)
    - [Version history](#version-history)
    - [Future plans](#future-plans)
    - [Acknowledgements](#acknowledgements)
    - [Author and copyright](#author-and-copyright)

<!-- /TOC -->

<a id="markdown-introduction" name="introduction"></a>
## Introduction

The header file `matrix.hpp` contains a class template for matrices, optimized for performance. Expression templates are used to automatically perform multiple element-wise operations using only one loop whenever possible. Optionally, multithreading can be turned on, to parallelize operations on large matrices. The implementation is completely stand-alone, compatible with any C++17-compliant compiler, with no external requirements or dependencies.

This class template is intended for use in cases where only basic matrix operations are needed, and the extended functionality and features of full-fledged C++ linear algebra libraries is not required. In such cases, this class template has the advantage of being simple, lightweight, and easy-to-use, while nonetheless supporting advanced performance-oriented features such as expression templates and multithreading "out of the box" for all relevant matrix operations.

The implementation of expression templates is straightforward and compact, utilizing C++14/17 features such as automatic return type and template argument deduction to avoid redundancies in the code. Multithreading is implemented at the lowest level, using a custom-made thread pool and the built-in C++ thread support library, rather than an API such as OpenMP. This allows the parallel algorithms to be specially tailored to matrix operations, for optimal performance. It also ensures that the code is lightweight and maximally portable, as it only uses standard C++17 without any additional dependencies.

<a id="markdown-features" name="features"></a>
## Features

* **Fast:**
    * Built from scratch with performance in mind, particularly for use in scientific computing.
    * Employs smart pointers for leak-safe, zero-overhead memory management.
    * Compact code reduces both compilation time and binary size.
    * Expression templates allow an arbitrary number of element-wise operations to be performed using only one loop, without creating any temporary objects as intermediate steps, for a significant performance boost.
    * Multithreading is fully supported by all operations that can benefit from it, and can provide a substantial performance boost, limited only by the number of available hardware threads. This option is especially suitable for high-performance computing systems.
* **Lightweight:**
    * Only ~370 lines of code, or ~630 lines with multithreading enabled (excluding comments and blank lines).
    * Single header file: simply `#include "matrix.hpp"`, and you're done.
    * Header-only: no need to install or build the library.
    * Self-contained: no external requirements or dependencies. Works with any C++17-compliant compiler.
* **Easy to use:**
    * Both expression templates and multithreading are utilized automatically behind the scenes. The user does not need to do anything manually, except optionally enable the multithreading and define the block size.
    * The code is thoroughly documented using Doxygen comments - not only the interface, but also the implementation, in case you would like to make modifications.
* **Available matrix operations:**
    * Addition: `+`, `+=`
    * Subtraction: `-`, `-=`
    * Negation: `-`
    * Multiplication by scalar: `*`, `*=`
    * Matrix multiplication: `*`, `*=`, `multiply_with_init()`
    * Transposition: `transpose()`
    * Trace: `trace()`
    * Fill matrix with scalar: `fill()`
    * Copy and move assignment: `=`
* **Additional features:**
    * Print the matrix to a stream, formatted to visually look like a matrix.
    * Generate random matrices based on any random distribution.
    * Supports types with an additive identity different from zero.

<a id="markdown-usage" name="usage"></a>
## Usage

<a id="markdown-including-the-library" name="including-the-library"></a>
### Including the library

To use the matrix class template, simply include the header file:

```cpp
#include "matrix.hpp"
```

The contents of the header file belong to the namespace `matrices`, so they will not interfere with anything in the global namespace. The matrix class template itself is called `matrices::matrix<T>`. It is recommended to employ a `using` statement:

```cpp
using matrices::matrix;
```

Now the class template can be accessed simply by typing `matrix<T>`.

<a id="markdown-template-parameters" name="template-parameters"></a>
### Template parameters

The template only has one parameter: `T`, the type of elements in the matrix. For example:

* `matrix<int>` is a matrix of integers.
* `matrix<double>` is a matrix of real (floating-point) numbers.
* `matrix<complex<double>>` is a matrix of complex floating-point numbers (using the header `<complex>`).

For brevity, one can employ a `using` statement to specialize to a specific type. For example, the following statement will allow using `md` instead of `matrix<double>`:

```cpp
using md = matrices::matrix<double>;
```

More generally, `T` can be any user-defined class that has addition, subtraction, negation, and multiplication overloaded.

<a id="markdown-constructors" name="constructors"></a>
### Constructors

Matrix objects can be constructed using several different constructors. All constructors will throw `matrices::zero_size` if the number of rows or columns given to the constructor is zero.

* `matrix<T> A(rows, cols)` constructs an uninitialized matrix `A`. For example:

```cpp
matrix<double> A(2, 2); // Creates a 2x2 real matrix.
```

* `matrix<T> A(rows, cols, scalar)` constructs a matrix `A` with all of its elements initialized to `scalar`. For example:

```cpp
matrix<double> A(2, 2, 0); // Creates a 2x2 real matrix with its elements initialized to zero.
```

* `matrix<T> A(dim, diagonal_array)` constructs a diagonal matrix `A` and initializes its diagonal using an array. It's up to you to ensure that the array has `dim` elements. For example:

```cpp
double diag[2] = {1, 2};
matrix<double> A(2, diag); // Creates a 2x2 real matrix with its diagonal initialized to {1, 2}.
```

* `matrix<T> A(dim, diagonal_list)` constructs a diagonal matrix `A` and initializes its diagonal using an `std::initializer_list`. Will throw `matrices::initializer_wrong_size` if the size of the list does not equal `dim`. For example:

```cpp
matrix<double> A(2, {1, 2}); // Creates a 2x2 real matrix with its diagonal initialized to {1, 2}.
```

* `matrix<T> B(A)` copy-constructs a matrix `B` with the same dimensions and elements as `A`. For example:

```cpp
matrix<double> A(2, {1, 2});
matrix<double> B(A); // B will also be a 2x2 real matrix with its diagonal initialized to {1, 2}.
```

* `matrix<T> A(rows, cols, init_list)` constructs a matrix `A` and initializes its elements using an `std::initializer_list` (in row-major order). Will throw `matrices::initializer_wrong_size` if the size of the list does not equal `rows * cols`. For example:

```cpp
matrix<double> A(2, 2, {1, 2, 3, 4}); // Creates a 2x2 real matrix with its first row initialized to {1, 2} and its second row to {3, 4}.
```

* To construct a matrix `A` and initialize its elements using an array, simply use the `matrix<T> A(rows, cols)` constructor to create an uninitialized matrix, and then execute the `from_array` member function. It's up to you to ensure that the array has `rows * cols` elements. For example:

```cpp
double elements[] = {1, 2, 3, 4};
matrix<double> A(2, 2);
A.from_array(elements); // Creates a 2x2 real matrix with its first row initialized to {1, 2} and its second row to {3, 4}.
```

<a id="markdown-overloaded-operators-and-member-functions" name="overloaded-operators-and-member-functions"></a>
### Overloaded operators and member functions

Matrix elements can be accessed as follows:

* `A(i, j)` is the element of the matrix `A` at row `i` and column `j`, **without** range checking.
* `A.at(i, j)` is the element of the matrix `A` at row `i` and column `j`, **with** range checking. Will throw `matrices:index_out_of_range` if the requested row or column index is out of range.

The following arithmetic operators are overloaded:

* Unary `-` for matrix negation: `-A`.
* Binary `+` and `-` for matrix addition and subtraction: `A + B - C`. Will throw `matrices::incompatible_sizes_add` if the matrices do not have the same number of rows and columns.
* Binary `*` for matrix multiplication by a scalar: `A * s`.
* Binary `*` for matrix multiplication by a matrix: `A * B`. Will throw `matrices:incompatible_sizes_multiply` if the number of columns in the first matrix is not the same as the number of rows in the second matrix.
* The binary assignment operators `+=`, `-=`, and `*=` are also defined.

The following member functions are available:

* `A.get_rows()` and `A.get_cols()` return the number of rows and columns in `A`.
* `A.fill(scalar)` changes all the elements of `A` to `scalar`.
* `A.trace()` returns the trace, i.e. the sum of elements on the diagonal of `A`. Will throw `matrices::not_square` if the matrix is not square.
* `A.transpose()` returns the transpose of `A`.

<a id="markdown-printing-matrices-to-streams" name="printing-matrices-to-streams"></a>
### Printing matrices to streams

The operator `<<` is overloaded, so for example, `std::cout << A` will print the matrix `A` to the standard output. The output will be formatted to visually look like a matrix when printed to the terminal. For example, this code:

```cpp
matrix<double> A(2, 2, {1, 22, 333, 4444});
std::cout << A;
```

will print out:

```none
(    1   22 )
(  333 4444 )
```

The character width of each element will be determined automatically to ensure that the matrix is aligned correctly. This works in a type-agnostic way, even if the elements are not numbers. If your matrix has elements of a user-defined type, make sure to overload `<<` for that type, so that the elements can be printed.

<a id="markdown-vectors" name="vectors"></a>
### Vectors

To create a vector, simply create a matrix with 1 row (for a row vector) or 1 column (for a column vector). For example, this code:

```cpp
matrix<double> A(2, 2, {1, 2, 3, 4});
matrix<double> v(2, 1, {5, 6});
std::cout << "A ="
            << A
            << "v ="
            << v
            << "A * v ="
            << A * v;
```

will multiply the matrix `A` by the vector `v`:

```none
A =
( 1 2 )
( 3 4 )
v =
( 5 )
( 6 )
A * v =
( 17 )
( 39 )
```

<a id="markdown-generating-random-matrices" name="generating-random-matrices"></a>
### Generating random matrices

The class template `matrix<T>` includes a nested class template called `random_generator<D>`, which allows generating random matrices using any random number distribution `D`. The constructor of `random_generator<D>` takes an arbitrary number of parameters, which are passed to the constructor of `D`.

For example, the following statement will construct an object called `rnd`, which will generate real matrices with random elements uniformly distributed between -10 and 10:

```cpp
matrix<double>::random_generator<std::uniform_real_distribution<double>> rnd(-10, 10);
```

In this case, the parameters were passed to the constructor of [std::uniform_real_distribution](https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution), which takes two parameters: the minimum and maximum values of the distribution. See [here](https://en.cppreference.com/w/cpp/numeric/random) for a full list of the available distributions and the parameters they accept.

Once a `random_generator` object has been created, we can use two equivalent member functions to generate random matrices:

* `generate_matrix(rows, cols)` returns a random matrix with the desired number of rows and columns. For example:

```cpp
matrix<double> A = rnd.generate_matrix(2, 2); // Creates a random 2x2 real matrix
```

* `randomize_matrix(A)` randomizes the matrix `A`, overwriting any existing elements with random elements. For example:

```cpp
matrix<double> A(2, 2);  // Creates an uninitialized 2x2 real matrix
rnd.randomize_matrix(A); // Randomizes the matrix A
```

In addition, `generate_scalar()` will return a random scalar from the distribution `D`.

<a id="markdown-types-with-an-additive-identity-different-from-zero" name="types-with-an-additive-identity-different-from-zero"></a>
### Types with an additive identity different from zero

Matrix multiplication `C = A * B`, as defined using the `*` operator, replaces each element `C(i, j)` with the inner product of row `i` of `A` with column `j` of `B`. The inner product is calculated by summing over the products of the elements, starting from zero. For user-defined types where the additive identity is not zero, use instead `C = multiply_with_init(A, B, init)` where `init` is the additive identity (similar to the third argument of `std::accumulate`).

Similarly, the `trace` member function accepts an optional parameter `init`, i.e. `A.trace(init)`, which will be used as the initial value of the sum.

<a id="markdown-degenerate-matrices" name="degenerate-matrices"></a>
### Degenerate matrices

Degenerate matrices, with zero rows and/or columns, will be created in two cases:

1. When using the default constructor, e.g. `matrix<T> A`.
2. When moving the contents of one matrix to another, e.g. `matrix<T> B(std::move(A))`, which will leave `A` as a degenerate matrix.

There is usually no reason to intentionally create a degenerate matrix, but it may be useful in some cases. For example, one may wish to create an array of degenerate matrices and construct each matrix to a specific size later. However, performing any operations on degenerate matrices, other than using the assignment operator, may have unpredictable results and/or throw `matrices::zero_size`.

<a id="markdown-expression-templates" name="expression-templates"></a>
## Expression templates

<a id="markdown-how-expression-templates-work" name="how-expression-templates-work"></a>
### How expression templates work

This library utilizes [expression templates](https://en.wikipedia.org/wiki/Expression_templates) to allow an arbitrary number of element-wise operations to be performed using only one loop, without creating any temporary objects as intermediate steps. This can result in a very significant performance boost. Expression templates are employed automatically at compilation time; you do not need to enable them manually.

The affected operations are:

* Addition
* Subtraction
* Negation
* Multiplication by scalar

For example, without expression templates, an expression such as `A = -B + 2 * C - D * 3` will be evaluated as follows:

1. Calculate `-B` by looping over the elements of `B`, negating each one, and storing the result in a temporary object `temp_1`.
2. Calculate `2 * C` by looping over the elements of `C`, multiplying each one by 2, and storing the result in a temporary object `temp_2`.
3. Calculate `D * 3` by looping over the elements of `D`, multiplying each one by 3, and storing the result in a temporary object `temp_3`.
4. Calculate `temp_1 + temp_2` by looping over the elements of `temp_1`, adding to each one the corresponding element of `temp_2`, and storing the result in a temporary object `temp_4`.
5. Calculate `temp_4 - temp_3` by looping over the elements of `temp_4`, subtracting from each one the corresponding element of `temp_3`, and storing the result in a temporary object `temp_5`.
6. Move the contents of `temp_5` into `A` and release the memory previously used by `A`.

Overall, we have five different loops, each taking time to complete, and five temporary objects, each taking up memory. In contrast, with expression templates, the same expression will be calculated with only one loop, and the result will be stored directly into `A`, without any intermediate steps or temporary objects:

```cpp
for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
        A(i, j) = -B(i, j) + 2 * C(i, j) - D(i, j) * 3
```

Therefore, the entire calculation will be 5 times faster and use 1/5 of the memory. The way this works is that whenever you perform any of the 4 element-wise operations mentioned above, the result will be not be a matrix, but rather an expression template (internally represented as an object of the class `matrix_op`) which simply records the calculation to be performed. The actual calculation, performing all of the recorded operations in one loop, will only take place in the following two cases:

1. When you use the `=` operator to assign an expression template to an existing matrix. For example, `A = B + 2 * C` will calculate `B + 2 * C` and store the result directly in `A` (in one loop and without any temporary objects).
2. When you pass an expression template as an argument to the constructor of a new matrix. For example, `matrix<double> A(B + 2 * C)` will construct `A` in place using the results of calculating `B + 2 * C`.
3. When you pass an expression template as an argument to the constructor of a temporary matrix object. For example, `matrix<double>(B + 2 * C)` will construct a temporary object in place using the results of calculating `B + 2 * C`. Such as object can then be passed to any function that accepts matrices - see below.

<a id="markdown-forcing-evaluation-of-expression-templates" name="forcing-evaluation-of-expression-templates"></a>
### Forcing evaluation of expression templates

Since an expression template such as `A + B` is merely a recipe for evaluation, you cannot use it directly with any function or operation that expects to act on a matrix and not an expression template. This includes row-wise operations such as multiplication and transposition. For example, the following statements will not compile:

```cpp
(A + B) * (C + D);
(A + B).transpose();
```

To execute these statements, simply wrap the expression templates inside a matrix constructor:

```cpp
matrix<double>(A + B) * matrix<double>(C + D);
matrix<double>(A + B).transpose();
```

Note that there is no performance penalty in doing so, as no copying is involved - the matrices will simply be constructed in place using the result of evaluating the expression templates, and will then be used directly by the multiplication and transposition functions.

The reason for this is that expression templates are evaluated element-wise and read memory sequentially, while operations such as multiplication and transposition are evaluated row-wise and read memory non-sequentially. Therefore, combining the former and the latter into one loop will hinder performance. This is especially significant in the case of multithreading, where element-wise and row-wise operations are parallelized in different ways. Therefore, row-wise functions expect to get matrices and not expression templates as arguments, and so you must explicitly evaluate the expression templates before passing them to the functions.

<a id="markdown-multithreading" name="multithreading"></a>
## Multithreading

<a id="markdown-enabling-multithreading" name="enabling-multithreading"></a>
### Enabling multithreading

To turn on multithreading, simply define the macro `ENABLE_MATRIX_MULTITHREADING`, **before** including the header file:

```cpp
#define ENABLE_MATRIX_MULTITHREADING
#include "matrix.hpp"
```

This enables automatic multithreading behind the scenes; no further configuration is necessary. Specifically, the following operations become multithreaded:

* Addition (including addition assignment)
* Subtraction (including subtraction assignment)
* Negation
* Multiplication by scalar (including multiplication assignment)
* Matrix multiplication (including multiplication assignment and `multiply_with_init`)
* Transposition
* Filling a matrix with a scalar
* Constructing a diagonal matrix using an array or an `std::initializer_list`
* Constructing a matrix using an `std::initializer_list`
* Constructing a matrix with all elements initialized to the same scalar
* Constructing a matrix from another matrix (or expression template)
* Assigning a matrix (or expression template) to another matrix
* Copying elements from an array to the matrix with `from_array`
* Generating a random matrix or randomizing an existing matrix

<a id="markdown-how-multithreading-works-and-when-to-use-it" name="how-multithreading-works-and-when-to-use-it"></a>
### How multithreading works, and when to use it

The library utilizes a custom-made thread pool to avoid the overhead of starting and joining individual threads. The threads are created when the program starts, and destroyed when the program exits. During the lifetime of the program, the threads will remain idle until you perform any of the multithreaded operations listed above.

When such an operation is performed, it is automatically parallelized into separate tasks, which are submitted to the thread pool's queue. Each of the idle threads in the pool will pick up a task from the queue, execute it, and then repeat the process until the queue is empty, at which point the thread will become idle again. Once all tasks have finished executing, control is given back to the main program.

Although using a thread pool eliminates the overhead of creating the individual threads, submitting tasks to the thread pool has its own (small) overhead. For this reason, using multithreading with small matrices may actually result in an overall performance penalty. Therefore, it is recommended to only enable multithreading if you use matrices large enough to benefit from parallelization.

The exact meaning of "large enough" varies based on the available hardware and the type of the elements. As a rule of thumb, for a `matrix<double>`, you should only enable multithreading if the matrices have at least 100 rows and columns. If you are not sure if your matrices are large enough, the best thing to do is to simply test the performance with and without multithreading, and see which option produces the best results.

As an example, consider adding two 4&times;4 matrices with 16 threads available. For a `matrix<double>`, parallelizing this operation would be a huge waste, since each parallel task will only be adding two numbers, and submitting the task to the pool will take longer than actually performing the addition. However, if the matrix elements are of a user-defined type for which addition is a very complicated and time-consuming calculation, then even though the matrix is very small, it would be strongly preferable to parallelize this operation.

Finally, note that, with multithreading disabled, the relevant portions of the code (such as the thread pool and parallelizer classes) are not included in the source code. This reduces the size of the included code roughly by half, and thus decreases both compilation time and executable size.

<a id="markdown-setting-a-custom-block-size" name="setting-a-custom-block-size"></a>
### Setting a custom block size

Whenever a parallelizable operation takes place, the block size determines how many elements will be operated on in parallel by each task. By default, the block size is automatically chosen on-the-fly such that each parallel operation is divided into a number of tasks equal to the number of available threads, which should provide near-optimal performance.

To manually set the block size, you may modify the static member variable `matrix<T>::global_block_size`. For example, if you have a 500&times;500 matrix, with 250,000 elements in total, and you set the block size to 10,000, then there will 25 blocks. Automatic choice of the block size can be restored by resetting this variable to the default value of zero.

Note that setting the block size is done separately for each type `T`, so a `matrix<double>` can have a different block size than a `matrix<int>`. Also, the value set by `matrix<T>::global_block_size` will only be applied to newly created matrices; the block size will not be changed retroactively for any existing matrices. When you create a matrix object, it will forever have the block size given by `matrix<T>::global_block_size` at that time.

It is very important to choose an appropriate block size for the task at hand. Too small blocks would mean too many tasks will be submitted to the thread pool, leading to a slowdown due to overhead in submitting each task. Too large blocks would mean not enough parallelization will take place. It is highly recommended to benchmark different block sizes in order to figure out the ideal block size for your program; see example below.

<a id="markdown-using-single-threaded-and-multithreaded-matrices-simultaneously" name="using-single-threaded-and-multithreaded-matrices-simultaneously"></a>
### Using single-threaded and multithreaded matrices simultaneously

In some cases, it may be desirable to have both single-threaded and multithreaded matrices in the same code - for example, if dealing with both very small and very large matrices. There are two methods to achieve this.

The first method is to set the block size manually so that the small matrices do not get parallelized. For example, if you have both 4&times;4 and 400&times;400 matrices, and 16 available threads, then setting the block size to 400&times;400/16 = 10,000 will ensure that operations on the large matrices are divided into one task per thread, while operations on the small matrices will only use one task and one thread.

The second method is to create two copies of `matrix.hpp`, e.g. `matrix_st.hpp` and `matrix_mt.hpp`, and change the name of the namespace in each file from `matrices` to e.g. `matrix_st` and `matrix_mt` respectively. Then both files may be included as follows:

```cpp
#include "matrix_st.hpp"
#define ENABLE_MATRIX_MULTITHREADING
#include "matrix_mt.hpp"
```

You will now be able to use `matrix_st::matrix<T>` for a single-threaded matrix and `matrix_mt::matrix<T>` for a multithreaded matrix, without having to specify the block size manually.

<a id="markdown-compiling" name="compiling"></a>
## Compiling

This library was tested on the following compilers and platforms:

* GCC v10.2.0 on Windows 10 build 19042.685 and Ubuntu 20.04.1 LTS.
* Clang 11.0.0 on Windows 10 build 19042.685 and Ubuntu 20.04.1 LTS.
* MSVC v14.28.29333 on Windows 10 build 19042.685.

Some notes on which compiler flags to use:

* As this library requires C++17 features, the code must be compiled with C++17 support. For GCC and Clang, use the `-std=c++17` flag. For MSVC, use `/std:c++17`.
* For MSVC, you must pass `/permissive-` to the compiler to enable standards conformance.
* If multithreading is enabled, you may need to pass `-pthread` to GCC and Clang on Linux.
* Of course, it is highly recommended to enable compiler optimizations (when not debugging). For GCC and Clang, `-O3` is recommended. For MSVC, `/O2` is recommended.

<a id="markdown-performance-tests" name="performance-tests"></a>
## Performance tests

The included file `matrix_test.cpp` both demonstrates how to use the matrix class template, and tests the multithreading performance of `matrix<double>` with various block sizes. On my computer, with a 12-core / 24-thread AMD Ryzen 9 3900X CPU, using GCC v10.2.0 on Windows 10 build 19042.685 with the `-O3` compiler flag, the results are as follows:

```none
Adding two 4800x4800 matrices (A = R + S):
With block size of 23040000 ( 1 blocks), execution took 63 ms.
With block size of  3840000 ( 6 blocks), execution took 26 ms.
With block size of  1920000 (12 blocks), execution took 24 ms.
With block size of   960000 (24 blocks), execution took 25 ms.
With block size of   480000 (48 blocks), execution took 24 ms.
With block size of   240000 (96 blocks), execution took 25 ms.

Adding three 4800x4800 matrices (A = R + S + T):
With block size of 23040000 ( 1 blocks), execution took 65 ms.
With block size of  3840000 ( 6 blocks), execution took 31 ms.
With block size of  1920000 (12 blocks), execution took 29 ms.
With block size of   960000 (24 blocks), execution took 28 ms.
With block size of   480000 (48 blocks), execution took 29 ms.
With block size of   240000 (96 blocks), execution took 28 ms.

Adding four 4800x4800 matrices (A = R + S + T + U):
With block size of 23040000 ( 1 blocks), execution took 65 ms.
With block size of  3840000 ( 6 blocks), execution took 33 ms.
With block size of  1920000 (12 blocks), execution took 33 ms.
With block size of   960000 (24 blocks), execution took 32 ms.
With block size of   480000 (48 blocks), execution took 32 ms.
With block size of   240000 (96 blocks), execution took 31 ms.

Adding four 4800x4800 matrices with scalar coefficients
(A = x * R + y * S + z * T + w * U):
With block size of 23040000 ( 1 blocks), execution took 70 ms.
With block size of  3840000 ( 6 blocks), execution took 37 ms.
With block size of  1920000 (12 blocks), execution took 33 ms.
With block size of   960000 (24 blocks), execution took 32 ms.
With block size of   480000 (48 blocks), execution took 32 ms.
With block size of   240000 (96 blocks), execution took 32 ms.

Generating random 4800x4800 matrix (rnd.randomize_matrix(A)):
With block size of 23040000 ( 1 blocks), execution took 297 ms.
With block size of  3840000 ( 6 blocks), execution took 52 ms.
With block size of  1920000 (12 blocks), execution took 29 ms.
With block size of   960000 (24 blocks), execution took 32 ms.
With block size of   480000 (48 blocks), execution took 26 ms.
With block size of   240000 (96 blocks), execution took 23 ms.

Transposing one 4800x4800 matrix (A = R.transpose()):
With block size of 23040000 ( 1 blocks), execution took 186 ms.
With block size of  3840000 ( 6 blocks), execution took 41 ms.
With block size of  1920000 (12 blocks), execution took 28 ms.
With block size of   960000 (24 blocks), execution took 26 ms.
With block size of   480000 (48 blocks), execution took 22 ms.
With block size of   240000 (96 blocks), execution took 20 ms.

Multiplying two 800x800 matrices (A = X * Y):
With block size of   640000 ( 1 blocks), execution took 414 ms.
With block size of   106666 ( 6 blocks), execution took 88 ms.
With block size of    53333 (12 blocks), execution took 49 ms.
With block size of    26666 (24 blocks), execution took 42 ms.
With block size of    13333 (48 blocks), execution took 31 ms.
With block size of     6666 (96 blocks), execution took 27 ms.
```

Here are some lessons we can learn from these results:

* For simple element-wise operations such as addition, multithreading improves performance very modestly, only by about a factor of 2, even when utilizing every available hardware thread. This is because compiler optimizations already parallelize simple loops fairly well on their own. Omitting the `-O3` flag, addition takes 230 ms with 1 block vs. 25 ms with 96 blocks, for a 9x speedup. However, you will most likely be compiling with optimizations turned on anyway.
* Thanks to expression templates, adding four matrices with scalar coefficients (7 operations) takes about as much time as adding two matrices (1 operation).
* Matrix multiplication and random matrix generation, which are more complicated operations that cannot be automatically parallelized by compiler optimizations, gain the most out of multithreading - with a very significant 13x to 15x speedup. Given that the test CPU only has 12 physical cores, and hyperthreading can generally produce no more than a 30% performance improvement, a 15x speedup is about as good as can be expected!
* Transposition also enjoys a 9x speedup with multithreading. Note that transposition requires reading memory is non-sequential order, jumping between the rows of the source matrix, which is why it's much slower, when single-threaded, than sequential operations such as addition.
* Even though the test CPU only has 24 threads, there is still a small but consistent benefit to dividing the work into 48 or even 96 parallel blocks. This is especially significant in multiplication, where we get a 50% speedup with 96 blocks (4 blocks per thread) compared to 24 blocks.

<a id="markdown-version-history" name="version-history"></a>
## Version history

* Version 1.0 (2021-01-03)
    * Initial release.

<a id="markdown-future-plans" name="future-plans"></a>
## Future plans

* Implement more complex numerical operations such as LU decomposition, inverse matrix, and diagonalization.
* Implement higher-rank tensors.
* Implement matrices of matrices (e.g. `matrix<matrix<double>>`). This is currently not possible, since it confuses the expression template system.

If you encounter any bugs, or would like a request any additional features, please feel free to open a new issue!

<a id="markdown-acknowledgements" name="acknowledgements"></a>
## Acknowledgements

I would like to thank [Erik Schnetter](https://github.com/eschnett) for his valuable advice regarding the thread pool implementation in this library.

<a id="markdown-author-and-copyright" name="author-and-copyright"></a>
## Author and copyright

Copyright (c) 2021 [Barak Shoshany](http://baraksh.com) (baraksh@gmail.com). Licensed under the [MIT license](LICENSE.txt).
