cmake_minimum_required(VERSION 3.16)
project(ShapeBasedMatcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (use same version as Emgu.CV 4.9.0)
find_package(OpenCV 4 REQUIRED)

# Enable SIMD optimizations
if(MSVC)
    add_compile_options(/arch:AVX2)
    add_compile_definitions(_USE_MATH_DEFINES)
else()
    add_compile_options(-mavx2 -mfma)
endif()

# Source files
set(SOURCES
    src/line2Dup.cpp
    src/shape_matcher_c_api.cpp
)

set(HEADERS
    src/line2Dup.h
    src/shape_matcher_c_api.h
)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/MIPP
    ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Export symbols on Windows
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SHAPEMATCHER_EXPORTS)
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Copy DLL to output for easier testing
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${PROJECT_NAME}>
            ${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>/
    )
endif()
