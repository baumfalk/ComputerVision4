cmake_minimum_required(VERSION 2.8.4)

project(LinearSVMFaceDetector)
SET(CMAKE_VERBOSE_MAKEFILE OFF)

if(NOT CMAKE_BUILD_TYPE)
    MESSAGE( STATUS "Please set the CMAKE_BUILD_TYPE environment variable accordingly for other builds than 'Debug'." )
    SET(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    if(CMAKE_COMPILER_IS_GNUCC)
        add_definitions(-O0)
        add_definitions(-g3)
    endif()
    add_definitions(-DDEBUG)
    set(Boost_DEBUG ON)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    if(CMAKE_COMPILER_IS_GNUCC)
        add_definitions(-O3)
        add_definitions(-fopenmp)
        add_definitions(-DPARALLEL_PROCESS)
    endif()
    add_definitions(-g3)
    add_definitions(-DDEBUG)
    set(Boost_DEBUG ON)
else()
    SET(CMAKE_BUILD_TYPE Release)
    if(CMAKE_COMPILER_IS_GNUCC)
        add_definitions(-O3)
        add_definitions(-fopenmp)
        add_definitions(-DPARALLEL_PROCESS)
    endif()
    add_definitions(-DNDEBUG)
    set(Boost_DEBUG OFF)
endif()

MESSAGE( STATUS "Build type: " ${CMAKE_BUILD_TYPE} ", Boost_DEBUG:" ${Boost_DEBUG})

add_definitions(-DTIXML_USE_TICPP)

if(CMAKE_COMPILER_IS_GNUCC)
    add_definitions(-D__GXX_EXPERIMENTAL_CXX0X__)
    add_definitions(-std=c++11)
endif()

find_package(OpenCV 2.4.6 REQUIRED)

# ========== BOOST ============
set( BOOST_COMPONENTS_NEEDED system filesystem regex )

# The following verifyies that BOOST_ROOT is set properly.
if(NOT BOOST_ROOT AND NOT $ENV{BOOST_ROOT} STREQUAL "")
    FILE( TO_CMAKE_PATH $ENV{BOOST_ROOT} BOOST_ROOT )
    if( NOT EXISTS ${BOOST_ROOT} )
        MESSAGE( STATUS  ${BOOST_ROOT} " does not exist. Checking if BOOST_ROOT was a quoted string.." )
        STRING( REPLACE "\"" "" BOOST_ROOT ${BOOST_ROOT} )
        if( EXISTS ${BOOST_ROOT} )
            MESSAGE( STATUS "After removing the quotes " ${BOOST_ROOT} " was now found by CMake" )
        endif( EXISTS ${BOOST_ROOT})
    endif( NOT EXISTS ${BOOST_ROOT} )

# Save the BOOST_ROOT in the cache
    if( NOT EXISTS ${BOOST_ROOT} )
        MESSAGE( WARNING ${BOOST_ROOT} " does not exist." )
    else(NOT EXISTS ${BOOST_ROOT})
        SET (BOOST_ROOT ${BOOST_ROOT} CACHE STRING "Set the value of BOOST_ROOT to point to the root folder of your boost install." FORCE)
    endif( NOT EXISTS ${BOOST_ROOT} )

endif(NOT BOOST_ROOT AND NOT $ENV{BOOST_ROOT} STREQUAL "")

if( WIN32 AND NOT BOOST_ROOT )
    MESSAGE( WARNING "Please set the BOOST_ROOT environment variable." )
endif( WIN32 AND NOT BOOST_ROOT )

set(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
FIND_PACKAGE(Boost 1.47.0 COMPONENTS ${BOOST_COMPONENTS_NEEDED})
if(Boost_FOUND)
    MESSAGE( STATUS "Setting up boost." )
    include_directories(${Boost_INCLUDE_DIRS})
    if(Boost_DEBUG)
        MESSAGE( STATUS "BOOST Libraries " ${Boost_LIBRARIES} )
        FOREACH(BOOST_COMPONENT ${BOOST_COMPONENTS_NEEDED})
            STRING( TOUPPER ${BOOST_COMPONENT} BOOST_COMPONENT_UPCASE )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} ": " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY} )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} " Debug: " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY_DEBUG} )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} " Release: " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY_RELEASE} )
        ENDFOREACH(BOOST_COMPONENT)
    endif(Boost_DEBUG)
else()
    MESSAGE ( WARNING ${Boost_LIBRARIES} )
endif(Boost_FOUND)
# ========== BOOST ============

#set the default path for built executables to the BUILD_TYPE directory ("Release" or "Debug")
if(WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
    MESSAGE ( STATUS "Bin output in: " ${EXECUTABLE_OUTPUT_PATH} ${CMAKE_BUILD_TYPE} )
else()
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
    MESSAGE ( STATUS "Bin output in: " ${EXECUTABLE_OUTPUT_PATH} )
endif()


include_directories (
    include
    modules
)

link_directories (
)

add_library (ticpp
    modules/ticpp/ticpp.cpp
    modules/ticpp/tinystr.cpp
    modules/ticpp/tinyxml.cpp
    modules/ticpp/tinyxmlerror.cpp
    modules/ticpp/tinyxmlparser.cpp
)

add_executable(${CMAKE_PROJECT_NAME}
    src/Detector.cpp
    src/controllers/FileIO.cpp
    src/controllers/QueryXML.cpp
    src/controllers/MySVM.cpp
    src/controllers/Utility.cpp
)

if(CMAKE_COMPILER_IS_GNUCC)
    target_link_libraries (${CMAKE_PROJECT_NAME} gomp)
endif()
target_link_libraries (${CMAKE_PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries (${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries (${CMAKE_PROJECT_NAME} ticpp)