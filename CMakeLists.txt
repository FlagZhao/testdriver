cmake_minimum_required(VERSION 3.10)
project(testdriver)

include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING
            "Default build type: DEBUG" FORCE)
endif()


SET(CMAKE_CXX_FLAGS "-O0")
SET(CMAKE_C_FLAGS "-O0")

if(${CMAKE_VERSION} VERSION_LESS "3.17.0") 
    find_package(CUDA REQUIRED)
    set(CUDAToolkit_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    set(CUDAToolkit_TARGET_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    if( ${CUDA_VERSION} LESS 11.2 )
        message(FATAL_ERROR "CUDAToolkit mush be newer or equal to 11.2. Your version is " ${CUDA_VERSION} )
    endif()

else()
    find_package(CUDAToolkit REQUIRED)
    if (NOT CMAKE_CUDA_COMPILER)
        if (NOT ${CUDAToolkit_NVCC_EXECUTABLE} AND UNIX AND EXISTS "/usr/local/cuda/bin/nvcc")
            set(CUDAToolkit_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
            message(WARNING "CMAKE_CUDA_COMPILER guessed: " ${CUDAToolkit_NVCC_EXECUTABLE} "\n"
                    "Please fix your cuda installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-post")
        endif ()
    endif ()
    if( ${CUDAToolkit_VERSION} LESS 11.2 )
        message(FATAL_ERROR "CUDAToolkit mush be newer or equal to 11.2. Your version is " ${CUDAToolkit_VERSION} )
    endif()
endif()


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CAMKE_CXX_FLAGS " -DSTANDALONE -O0")

set(SANITIZER_PATH ${CUDAToolkit_TARGET_DIR}/compute-sanitizer)

include_directories(${SANITIZER_PATH}/include/ 
                ${CUDAToolkit_TARGET_DIR}/include/ 
                ${CMAKE_CURRENT_LIST_DIR}/include/
                ${CMAKE_CURRENT_LIST_DIR}/src/
                )
                
message(INFO ${SANITIZER_PATH} ${CUDAToolkit_NVCC_EXECUTABLE})

add_link_options(-L${SANITIZER_PATH} -fPIC -shared -lstdc++)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/ testdriver_SRC)

file(GLOB testdriver_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/include/*.h" )
add_library(testdriver SHARED ${testdriver_SRC} ${testdriver_headers} )


target_link_libraries(testdriver PUBLIC
        
        -lsanitizer-public
        -lcuda
        -lcudart
        -L${CUDAToolkit_TARGET_DIR}/lib64
        -ldl
        )



install(TARGETS testdriver DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES  ${testdriver_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
