add_triton_library(TritonCPUXsmm
    ConvertVectorToXsmm.cpp
    VnniUtils.cpp
    ValueUtils.cpp
    XsmmEnum.cpp
    XsmmUtils.cpp

    DEPENDS
    TritonCPUXsmmPassIncGen
    TritonCPUXsmmAttrDefIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRVectorDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
    MLIRLinalgUtils
)
