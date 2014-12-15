cmake_minimum_required(VERSION 2.8)
#-------------------------------------------------------------------
#the name of the executable will be Project
#-------------------------------------------------------------------
project(Project)

#-------------------------------------------------------------------
#set directories
#-------------------------------------------------------------------
if (WIN32)
#set (OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../3rdParty/OpenCV/build)
set (OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../3rdParty/OpenCV/sources-build)
set (Eigen_DIR ${CMAKE_SOURCE_DIR}/../../3rdParty/Eigen)
set (OpticalFlow_DIR ${CMAKE_SOURCE_DIR}/../../3rdParty/OpticalFlow)
endif (WIN32)


#-------------------------------------------------------------------
#collect source files
#-------------------------------------------------------------------
set(HEADER_DIR ${CMAKE_SOURCE_DIR}/header)
include_directories(${HEADER_DIR})
file(GLOB_RECURSE SOURCES *.cpp)
file(GLOB_RECURSE HEADER *.h)
file(GLOB_RECURSE CUDASRC *.cu)


#-------------------------------------------------------------------
#find opencv
#-------------------------------------------------------------------
find_package( OpenCV REQUIRED )
if(OPENCV_FOUND)
    message(STATUS "OpenCV found")
else()
    message(STATUS "OpenCV not found")
endif(OPENCV_FOUND)


#-------------------------------------------------------------------
#find cuda
#-------------------------------------------------------------------
FIND_PACKAGE(CUDA REQUIRED)
set (CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
#HW capability 1.3

SET(LIB_TYPE STATIC)	#set the lib type
CUDA_ADD_LIBRARY(CudaLib ${LIB_TYPE} ${CUDASRC})


#-------------------------------------------------------------------
#add include directories
#-------------------------------------------------------------------
include_directories(${Eigen_DIR}) #Eigen is header only
include_directories(${OpticalFlow_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/kernels)


#-------------------------------------------------------------------
#add source and headers
#-------------------------------------------------------------------
cuda_add_executable(Project ${SOURCES} ${HEADER})


#-------------------------------------------------------------------
#link all libraries to the project
#-------------------------------------------------------------------
ADD_DEPENDENCIES(Project CudaLib)
target_link_libraries(Project ${OpenCV_LIBS} CudaLib)


#-------------------------------------------------------------------
#add a target to generate API documentation with Doxygen
#-------------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
	${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)


#-------------------------------------------------------------------
#copy all dlls from OpenCV to this projects exe directory
#-------------------------------------------------------------------
#file(GLOB OPENCV_DLLS ${OpenCV_DIR}/x64/vc12/bin/*.dll)
file(GLOB OPENCV_DLLS ${OpenCV_DIR}/bin/*.dll)
file(COPY ${OPENCV_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
