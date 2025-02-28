**Important Note:** This project is an **AI project** and is intended for use only by **experienced professionals in artificial intelligence and deep learning**. It requires a strong understanding of these domains. 

⚙️ **Prerequisites**

Prior to building this project, ensure your development environment is equipped with the following:

- **Operating System**: Windows (tested on Windows 10/11). Other platforms may require adjustments.
- **Visual Studio 2022**: Download and install the latest version of Visual Studio 2022, including C++ development tools. ([Visual Studio](https://visualstudio.microsoft.com/))
- **HIP SDK**: The HIP SDK enables building applications for AMD GPUs. ([HIP SDK](https://github.com/ROCm-Software/HIP))
- **CUDA Toolkit**: For building CUDA-accelerated components, install CUDA Toolkit version 12.5 from NVIDIA. ([CUDA Toolkit](https://developer.nvidia.com/cuda-downloads))
- **cuDNN**: For optimized deep learning operations, install cuDNN version 9.2.1 compatible with CUDA 12.5. ([cuDNN](https://developer.nvidia.com/cudnn))

**Note:** Compatibility and specific versions of these components may vary. Consult the official documentation for the most up-to-date requirements.

📦 **Dependency Management**

### Submodules

Our project uses Git submodules to manage its dependencies. Follow these steps to initialize and update the submodules:

1. Initialize the submodules:
    ```bash
    git submodule init
    ```

2. Update the submodules:
    ```bash
    git submodule update
    ```

3. Download the dependencies into the `3rdparty` folder by running:
    ```bash
    python setup.py
    ```

4. Link the dependencies in Visual Studio 2022 as needed.

The following submodules are included:

* **cutlass:** CUDA C++ Templates for High Performance Matrix Multiplications. 
* **json:** A C++ JSON library for modern C++. 
* **cxxopts:** Lightweight C++ command-line option parser. 
* **NVTX:** NVIDIA Tools Extension library for profiling. 
* **OpenBLAS:** Optimized BLAS library. 
* **pybind11:** Seamless operability between C++ and Python. 
* **netcdf-cxx4:** C++ library for NetCDF. 
* **TensorRT:** NVIDIA TensorRT for high-performance inference. 
* **MPI:** Microsoft MPI library. 
* **cpython:** Python interpreter codebase. 
* **nccl:** NVIDIA Collective Communications Library. 
* **pytorch:** PyTorch deep learning framework. 
* **cudnn-frontend:** cuDNN Frontend for easier integration with cuDNN. 
* **jsoncpp:** JSON parser and serializer for C++. 
* **highway:** SIMD library for performance. 
* **googletest:** Google Test framework for C++. 
* **pthread-win32:** PThreads library for Windows. 
* **boost:** Boost C++ Libraries. 
* **glog:** Google Logging Library. 
* **cccl:** NVIDIA's Communication Concurrency and Coordination Library. 

**Installation:**

1. Ensure that all dependencies are installed on your system and their respective libraries are available in the system PATH.
2. Consult individual library documentation for specific installation procedures and requirements for your operating system.

🔨 **Building suggestify**

1. **Open Visual Studio 2022:** Launch Visual Studio 2022.
2. **Open Solution File:** Navigate to the suggestify project directory and open the `suggestify.sln` solution file.
3. **Select Build Configuration:** In the Visual Studio menu, choose the desired build configuration (e.g., Debug or Release).
4. **Build Solution:** Utilize the shortcut Ctrl + Shift + B to build the entire solution. Alternatively, use the "Build" menu and select "Build Solution".

**Note:** The build process can take up to 30 minutes.

🧪 **Running Tests**

Once the project has been built successfully, execute unit tests to verify its functionality:

1. **Navigate to the Tests Directory:** Open the `tests` folder within the suggestify project directory.
2. **Run Tests:** Locate the appropriate test executable (e.g., `suggestify_tests.exe`) and execute it.

**Note:** Ensure the required test dependencies are installed and the test environment is correctly configured.

📄 **Checking Logs**

You can view the log file under `Debug/HIP_nvcc/suggestify.log`. This log is useful for identifying bugs and other issues.
