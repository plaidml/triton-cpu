add_subdirectory(kernels)

get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_triton_library(TritonCPUXsmm
    ConvertTritonToXsmm.cpp
    ConvertVectorToXsmm.cpp
    VnniUtils.cpp
    ValueUtils.cpp
    XsmmEnum.cpp
    XsmmUtils.cpp

    DEPENDS
    TritonCPUXsmmPassIncGen
    TritonCPUXsmmAttrDefIncGen
    xsmm
    TritonCPUXsmmKernelUtils

    LINK_LIBS PUBLIC
    ${extension_libs}
    MLIRIR
    MLIRPass
    MLIRVectorDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
    MLIRLinalgUtils
    TritonCPUIR
    xsmm
    TritonCPUXsmmKernelUtils
)

target_include_directories(TritonCPUXsmm
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
)
